"""Importance-matrix calibration corpus builder.

Mix (master plan §4.8): 60% domain manuals / 25% replay chat / 15% raw QA,
~120k words, seed 42, holdout documents excluded (enforced here, not by
discipline). The output text file feeds `llama-imatrix -m <F16> -f <out>`.

Run:  python -m sahel_sage.training.imatrix --out <path>
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from sahel_sage.core.config import repo_root
from sahel_sage.core.textproc import iter_doc_chunks


def load_holdout() -> set[str]:
    data = json.loads((repo_root() / "data/splits/holdout.json").read_text())
    return set(data["doc_ids"])


def _replay_texts(path: Path) -> list[str]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if "text" in rec:
            out.append(rec["text"])
        else:
            out.append("\n".join(m["content"] for m in rec["messages"]))
    return out

def build_calibration(
    out_path: Path,
    corpus_dir: Path,
    replay_dir: Path,
    target_words: int = 120_000,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    holdout = load_holdout()

    domain = [
        text
        for _, text in iter_doc_chunks(
            corpus_dir, target_words=350, overlap_words=0, min_words=50, exclude_docs=holdout
        )
    ]
    chat = _replay_texts(replay_dir / "smoltalk.jsonl")
    raw_qa = _replay_texts(replay_dir / "arc_train.jsonl")
    for pool in (domain, chat, raw_qa):
        rng.shuffle(pool)

    budgets = {
        "domain": (domain, int(target_words * 0.60)),
        "chat": (chat, int(target_words * 0.25)),
        "raw_qa": (raw_qa, int(target_words * 0.15)),
    }
    parts: list[str] = []
    stats: dict[str, int] = {}
    for name, (pool, budget) in budgets.items():
        used = 0
        for item in pool:
            if used >= budget:
                break
            parts.append(item)
            used += len(item.split())
        stats[name] = used

    rng.shuffle(parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(parts) + "\n")
    stats["total"] = sum(stats.values())
    stats["holdout_excluded"] = len(holdout)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=repo_root() / "training/corpus_txt")
    ap.add_argument("--replay", type=Path, default=repo_root() / "training/replay")
    ap.add_argument("--target-words", type=int, default=120_000)
    args = ap.parse_args()
    stats = build_calibration(args.out, args.corpus, args.replay, args.target_words)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
