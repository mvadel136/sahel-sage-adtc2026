"""Numeric-invention gate: drop any training answer that states a number it was not given.

Round 4, asked for a pesticide rate, produced *"if you need 100 ml of pesticide
per litre…"*, a fabricated concentration, despite a system prompt forbidding
exactly that. A model that invents doses is a harm risk, not a scoring problem.

The chosen fix is filtering, not preference tuning. Preference tuning on
near-identical pairs ("same answer, with and without an invented number") is
the documented worst case for likelihood displacement: DPO **reduced refusal
rates from 80.5%→54.8% and 74.4%→33.4%** in arXiv 2410.08847 precisely when
chosen and rejected responses share embeddings. Filtering cannot backfire that
way, the model simply never sees an invented number.

Rule: a quantity in an answer must appear in that answer's source text.
Numbers with no source are dropped, and the record is reported so the gate's
strictness is visible rather than silent.

Deliberately NOT flagged: bare ordinals and small counts used structurally
("step 2", "3 weeks"), because time and counts are usually reasoning the model
should keep. The gate targets *measured quantities*, the ones that hurt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

#: Measured quantities: a number bound to a unit of mass, volume, area, rate or
#: concentration. These are the claims that can poison a field or an animal.
QUANTITY = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(ml|millilitres?|milliliters?|cl|l\b|litres?|liters?"
    r"|mg|g\b|grams?|kg|kilograms?|tonnes?|tons?"
    r"|%|percent"
    r"|ppm|cc\b"
    r"|(?:kg|g|l|ml|litres?)\s*(?:/|per\s+)(?:ha|hectare|litre|liter|l\b|m2|m²|plant|animal|head|bag)"
    r"|(?:/|per\s+)(?:ha|hectare)"
    r")",
    re.IGNORECASE,
)

#: Number-like strings that are fine anywhere: they describe structure or time,
#: not a measured dose.
_BENIGN = re.compile(
    r"\b\d+\s*(?:days?|weeks?|months?|years?|times?|hours?|minutes?"
    r"|cm|centimetres?|centimeters?|m\b|metres?|meters?)\b",
    re.IGNORECASE,
)


def quantities(text: str) -> list[str]:
    """Measured quantities mentioned in `text`, normalised for comparison."""
    cleaned = _BENIGN.sub(" ", text)
    return [f"{m.group(1).replace(',', '.')}{m.group(2).lower().replace(' ', '')}"
            for m in QUANTITY.finditer(cleaned)]


def _numbers_in(text: str) -> set[str]:
    return {m.group(1).replace(",", ".") for m in QUANTITY.finditer(text)}


def unsupported_quantities(answer: str, source: str) -> list[str]:
    """Quantities in `answer` whose numeric value never appears in `source`."""
    if not answer:
        return []
    src_numbers = _numbers_in(source or "")
    bad = []
    for m in QUANTITY.finditer(_BENIGN.sub(" ", answer)):
        value = m.group(1).replace(",", ".")
        if value not in src_numbers:
            bad.append(m.group(0).strip())
    return bad


def gate_records(
    records: Iterable[dict],
    source_for: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    """Split records into (kept, stats), dropping unsupported quantities.

    `source_for` maps passage_id -> chunk text. A record with no resolvable
    source is judged against an empty source, so ANY measured quantity in a
    closed-book answer is unsupported by construction, which is the intent:
    closed-book answers must not state doses.
    """
    source_for = source_for or {}
    kept: list[dict] = []
    stats = {"in": 0, "kept": 0, "dropped": 0, "examples": []}
    for rec in records:
        stats["in"] += 1
        answer = rec.get("a") or ""
        if not answer:
            kept.append(rec)
            stats["kept"] += 1
            continue
        meta = rec.get("meta", {})
        # `gate_passages` is the origin of a derived row (closed_book/bare):
        # its quantity was legitimately sourced even though the row itself
        # shows no evidence. Without this, deriving a row would turn a sourced
        # number into an "invented" one.
        pids = meta.get("gate_passages") or meta.get("passage_ids") or []
        source = " ".join(source_for.get(pid, "") for pid in pids)
        bad = unsupported_quantities(answer, source)
        if bad:
            stats["dropped"] += 1
            if len(stats["examples"]) < 15:
                stats["examples"].append({"id": rec.get("id"), "quantities": bad})
            continue
        kept.append(rec)
        stats["kept"] += 1
    return kept, stats


def gate_file(
    in_path: Path,
    out_path: Path,
    chunks_path: Path | None = None,
) -> dict:
    source_for: dict[str, str] = {}
    if chunks_path and chunks_path.exists():
        for line in chunks_path.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                source_for[c["chunk_id"]] = c["text"]
    records = [json.loads(x) for x in in_path.read_text().splitlines() if x.strip()]
    kept, stats = gate_records(records, source_for)
    with out_path.open("w") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return stats
