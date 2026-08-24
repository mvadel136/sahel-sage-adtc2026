"""Retrieval quality + confidence-threshold calibration.

Ground truth comes free: every distilled pair records the passage it was
written from, so `pairs_en_normalized.jsonl` is a labelled query set of
thousands of realistic farmer questions with a known correct document. No
hand-written relevance set, no teacher calls, no annotator bias.

Two things are measured:

1. **Retrieval quality** — recall@k and MRR against the source document.
2. **Threshold calibration** — the distribution of EvidencePack.confidence for
   IN-corpus questions (should be answerable) vs OUT-of-corpus questions
   (agriculture-shaped but unanswerable, plus off-domain). The shipped
   threshold is chosen from the sweep that maximises Youden's J
   (sensitivity + specificity - 1), then reported with its full confusion
   matrix so the trade-off is explicit rather than assumed.

Run: python -m sahel_sage.evaluation.retrieval_eval --db app/library.db
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from sahel_sage.core.config import repo_root
from sahel_sage.retrieval.evidence import build_pack
from sahel_sage.retrieval.store import open_retriever

# Agriculture-shaped questions with no answer in a Sahel manual library:
# regional/temporal specifics, non-African crops, market data, machinery brands.
OUT_OF_CORPUS = [
    "What is the price of a 50 kg bag of urea in Nouakchott market today?",
    "Which fertiliser subsidy programme is open in Mauritania this month?",
    "How do I grow wasabi in a greenhouse?",
    "What is the pollen viability of Vitis vinifera in Bordeaux?",
    "How many hectares of quinoa were planted in Peru last year?",
    "Where is the nearest John Deere dealership to Rosso?",
    "What was the rainfall in Kaedi in March 2019?",
    "How do I get a bank loan to buy a tractor?",
    "Which cooperative should I join in Trarza?",
    "What is the export tariff for gum arabic to the European Union?",
    "How do I register my farm with the ministry?",
    "When will the next agricultural census take place?",
    "How much does a veterinary consultation cost in Nouadhibou?",
    "What is the phone number of the extension office?",
    "How do I grow maple syrup trees?",
    "What is the best snow cover for winter wheat in Canada?",
    "Which pesticide is approved by the US EPA for use on almonds?",
    "How do I set up drip irrigation using a smartphone app?",
    "What is the current dollar exchange rate for selling my millet?",
    "Who won the agricultural innovation prize last year?",
]


@dataclass
class Row:
    question: str
    expected_doc: str | None
    rank: int | None      # 1-based rank of the expected doc, None if absent
    confidence: float
    sufficient: bool
    in_corpus: bool


def _load_labelled(pairs_path: Path, n: int, seed: int) -> list[tuple[str, str]]:
    recs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(recs)
    out = []
    for r in recs:
        docs = r.get("meta", {}).get("source_docs") or []
        if docs and r.get("q"):
            out.append((r["q"], docs[0]))
        if len(out) >= n:
            break
    return out


def evaluate(db_path: Path, pairs_path: Path, k: int = 4, n: int = 300, seed: int = 42) -> dict:
    retriever = open_retriever(db_path)
    rows: list[Row] = []

    for q, doc in _load_labelled(pairs_path, n, seed):
        cites = retriever.search(q, k=k)
        cov = retriever.coverage_for(q, cites)
        pack = build_pack(cites, threshold=0.0, k=k, coverage=cov)
        rank = next((i for i, c in enumerate(cites, 1) if c.doc_id == doc), None)
        rows.append(Row(q, doc, rank, pack.confidence, pack.sufficient, True))

    for q in OUT_OF_CORPUS:
        cites = retriever.search(q, k=k)
        cov = retriever.coverage_for(q, cites)
        pack = build_pack(cites, threshold=0.0, k=k, coverage=cov)
        rows.append(Row(q, None, None, pack.confidence, pack.sufficient, False))

    in_rows = [r for r in rows if r.in_corpus]
    hits = [r for r in in_rows if r.rank is not None]
    quality = {
        "n_in_corpus": len(in_rows),
        f"recall@{k}": round(len(hits) / len(in_rows), 4) if in_rows else 0.0,
        "recall@1": round(sum(r.rank == 1 for r in hits) / len(in_rows), 4) if in_rows else 0.0,
        "mrr": round(sum(1 / r.rank for r in hits) / len(in_rows), 4) if in_rows else 0.0,
    }

    # threshold sweep: positive = in-corpus question that retrieval SHOULD serve
    sweep = []
    pos = [r.confidence for r in in_rows if r.rank is not None]
    neg = [r.confidence for r in rows if not r.in_corpus]
    for t in [i / 100 for i in range(0, 101, 2)]:
        tp = sum(c >= t for c in pos)
        fn = len(pos) - tp
        fp = sum(c >= t for c in neg)
        tn = len(neg) - fp
        sens = tp / (tp + fn) if pos else 0.0
        spec = tn / (tn + fp) if neg else 0.0
        sweep.append({"threshold": t, "sensitivity": round(sens, 4),
                      "specificity": round(spec, 4), "youden_j": round(sens + spec - 1, 4),
                      "tp": tp, "fn": fn, "fp": fp, "tn": tn})
    best = max(sweep, key=lambda s: (s["youden_j"], s["sensitivity"]))

    def stats(vals: list[float]) -> dict:
        if not vals:
            return {}
        s = sorted(vals)
        return {"min": round(s[0], 4), "p25": round(s[len(s) // 4], 4),
                "median": round(s[len(s) // 2], 4), "p75": round(s[3 * len(s) // 4], 4),
                "max": round(s[-1], 4)}

    return {
        "quality": quality,
        "confidence_in_corpus": stats(pos),
        "confidence_out_of_corpus": stats(neg),
        "recommended_threshold": best,
        "sweep": sweep,
        "misses": [r.question for r in in_rows if r.rank is None][:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=repo_root() / "app/library.db")
    ap.add_argument("--pairs", type=Path,
                    default=repo_root() / "training/distilled/pairs_en_normalized.jsonl")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    result = evaluate(args.db, args.pairs, k=args.k, n=args.n)
    printable = {kk: vv for kk, vv in result.items() if kk != "sweep"}
    print(json.dumps(printable, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
