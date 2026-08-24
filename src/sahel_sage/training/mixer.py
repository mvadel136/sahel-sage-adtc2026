"""Dataset mixer: merge distilled + replay (+ Wolof) into an immutable dataset-vN.

Guards enforced in code, not discipline:
- holdout leakage: any example touching a holdout doc aborts the build
- dedup: normalized-question prefix + exact-answer hash across ALL strata
- provenance: every record keeps its source doc/passage ids and teacher
- critique: only critique == "pass" records enter the mix

Output: datasets/mix/dataset-<version>/{train.jsonl, stats.json, manifest.json}
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from sahel_sage.core.config import repo_root
from sahel_sage.data.splits import assert_no_holdout, load_holdout

_WS = re.compile(r"\s+")

#: Strata that teach ONE answer through many questions, and so are exempt from
#: answer-dedup. Both exist to move a behaviour the model resists: the reference
#: topics override a pretraining prior ("plant millet in early spring" survived
#: two rounds of correction), and the prohibitions have to hold against a
#: question actively arguing the other way. Repetition is the mechanism, not an
#: accident, and paraphrasing the answer to satisfy the deduper would teach the
#: model that the answer is negotiable.
ONE_ANSWER_KINDS = frozenset({"reference_topic", "prohibition"})


def _norm_q(q: str) -> str:
    return _WS.sub(" ", q.lower().strip())[:160]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "absent"


def _add_replay(records: list[dict], stats: Counter, replay_dir: Path,
                rng: random.Random, replay_chat_cap: int) -> None:
    """General-ability replay. Kept working, but no longer mixed by default."""
    for rec in _read_jsonl(replay_dir / "arc_train.jsonl"):
        records.append({"kind": "replay_arc", "text": rec["text"],
                        "meta": rec.get("meta", {})})
        stats["replay_arc"] += 1
    chat = [
        rec
        for rec in _read_jsonl(replay_dir / "smoltalk.jsonl")
        if "\\boxed" not in json.dumps(rec) and "$" not in rec["messages"][0]["content"]
    ]
    rng.shuffle(chat)
    for rec in chat[:replay_chat_cap]:
        records.append(
            {"kind": "replay_chat", "messages": rec["messages"],
             "meta": rec.get("meta", {})}
        )
        stats["replay_chat"] += 1


def build_dataset(
    version: str,
    distilled_paths: list[Path],
    replay_dir: Path,
    wolof_path: Path | None = None,
    out_root: Path | None = None,
    seed: int = 42,
    replay_chat_cap: int = 1500,
    include_replay: bool = True,
    grounded_cap: int | None = None,
) -> dict:
    rng = random.Random(seed)
    holdout = load_holdout()
    out_dir = (out_root or repo_root() / "datasets" / "mix") / f"dataset-{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    stats: Counter = Counter()

    # --- distilled QA (grounded/closed-book/abstention/safety) ---
    for path in distilled_paths:
        for rec in _read_jsonl(path):
            if rec.get("meta", {}).get("critique") != "pass":
                stats["skip_critique"] += 1
                continue
            records.append(rec)
            stats[rec.get("kind", "unknown")] += 1

    # --- cap the grounded stratum -------------------------------------------
    # grounded_chunk was 6,637 rows but 55.3% of ALL training tokens, because
    # each carries retrieved passages. It is the single largest cost in every
    # round, and it trains an input format the judged path never produces:
    # a judge chats with the bare GGUF, which has no retrieval.
    #
    # Two primary sources (arXiv:2312.05934, arXiv:2402.05119) find small models
    # do not absorb facts by fine-tuning, and our own recall probe measured 5.6%
    # on exactly this content — so the round is paying its largest bill for
    # knowledge the model is not keeping. Facts belong in the reference block,
    # which ships in the template, and in the app's retrieval index. Neither is
    # touched by this cap; nothing is deleted, and raising the number rebuilds
    # the old mixture in ten minutes with no GPU.
    if grounded_cap is not None:
        grounded = [r for r in records if r.get("kind") == "grounded_chunk"]
        if len(grounded) > grounded_cap:
            keep = set(id(r) for r in rng.sample(grounded, grounded_cap))
            records = [r for r in records
                       if r.get("kind") != "grounded_chunk" or id(r) in keep]
            stats["grounded_capped"] = len(grounded) - grounded_cap
            stats["grounded_chunk"] = grounded_cap

    # --- wolof (only reviewed pairs reach this file by construction) ---
    if wolof_path is not None:
        for rec in _read_jsonl(wolof_path):
            rec.setdefault("kind", "wolof")
            records.append(rec)
            stats["wolof"] += 1

    # --- replay: arc raw completions (all) + chat (capped, math filtered) ---
    #
    # Off by default from v7. Replay exists to stop a fine-tune forgetting
    # general ability, measured on arc_easy and instruction-formatting — and
    # neither is scored here. The competition's accuracy half is five human
    # conversations with the bare GGUF.
    #
    # It was not merely neutral. `replay_arc` rows are "Question: ...\nAnswer:"
    # with ~21-character completions and no system prompt, so 1,100 rows were
    # teaching a terse factoid distribution against a contract asking for
    # 150-250 words in four named sections — and they did it in an input format
    # the judged path never produces. Together with replay_chat that was 13.6%
    # of the mix spent training away from the target behaviour.
    if include_replay:
        _add_replay(records, stats, replay_dir, rng, replay_chat_cap)

    # --- holdout guard (abort, never warn) ---
    touched = [
        doc
        for rec in records
        for doc in rec.get("meta", {}).get("source_docs", [])
    ]
    assert_no_holdout(touched, holdout)

    # --- dedup across all strata ---
    # Dedup is scoped by kind: the derived strata intentionally reuse grounded
    # questions (same question, different evidence context -> different correct
    # behaviour), so cross-kind question collisions are counter-training, not
    # duplication.
    seen_q: set[str] = set()
    seen_a: set[str] = set()
    kept: list[dict] = []
    for rec in records:
        kind = rec.get("kind", "unknown")
        # Strata where one answer is deliberately taught many ways. Answer-dedup
        # would keep exactly one row per topic or rule — it collapsed 300
        # reference rows to 15 and 420 prohibition rows to 7 on the first v6
        # build, the same way `abstain_scope` went 450 -> 81 -> 12 -> 1 across
        # three earlier rounds before anyone noticed.
        #
        # Repetition IS the mechanism here. These facts have to override a
        # pretraining prior, and a refusal has to hold against a question openly
        # arguing for the opposite; varying the answer to satisfy the deduper
        # would teach the model that the answer is negotiable. Questions are
        # still deduped, so the variety lives where it belongs.
        if kind in ONE_ANSWER_KINDS:
            q_key = f"{kind}|{_norm_q(rec['q'])}"
            if q_key in seen_q:
                stats["dedup_q"] += 1
                continue
            seen_q.add(q_key)
            kept.append(rec)
            continue
        # raw_text rows have no q/a; dedup them on the text itself so a
        # repeated chunk cannot be silently duplicated into the mix.
        if kind == "raw_text":
            q_key = f"raw|{hashlib.sha1(rec['text'].strip().encode()).hexdigest()}"
            a_key = None
        else:
            q_key = f"{kind}|{_norm_q(rec['q'])}" if "q" in rec else None
            a_key = (
                f"{kind}|{hashlib.sha1(rec['a'].strip().encode()).hexdigest()}"
                if "a" in rec
                else None
            )
        if q_key is not None and q_key in seen_q:
            stats["dedup_q"] += 1
            continue
        if a_key is not None and a_key in seen_a:
            stats["dedup_a"] += 1
            continue
        if q_key is not None:
            seen_q.add(q_key)
        if a_key is not None:
            seen_a.add(a_key)
        kept.append(rec)

    rng.shuffle(kept)
    train_path = out_dir / "train.jsonl"
    with train_path.open("w") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats_out = {"version": version, "total": len(kept), "strata": dict(stats), "seed": seed}
    (out_dir / "stats.json").write_text(json.dumps(stats_out, indent=2) + "\n")
    manifest = {
        "version": version,
        "inputs": {str(p): _sha(p) for p in distilled_paths},
        "replay_dir": str(replay_dir),
        "wolof": str(wolof_path) if wolof_path else None,
        "holdout_ids": sorted(holdout),
        "seed": seed,
        "train_sha256": _sha(train_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return stats_out
