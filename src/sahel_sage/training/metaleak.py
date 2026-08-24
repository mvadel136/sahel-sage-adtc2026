"""Reject answers that talk about their own prompt instead of answering.

Asked whether a washed pesticide container could carry drinking water, the model
opened with:

    "The answer is unclear, and no verified reference is given. From what is
     normally recommended: ..."

To a judge that is worse than the wrong answer underneath it. It breaks the
persona, narrates the scaffolding, and tells the reader that the advice they are
about to get is the model's second choice. A farmer does not know what a
"verified reference" is and should never have to.

The failure is inherited, not invented: earlier datasets contain grounded rows
whose answers legitimately say *"the extracts do not cover this"*, because on
the app path the model really is looking at numbered extracts. Once those
phrasings leak into closed-book rows — where there are no extracts and the
reader can see none — the model has learned to reference a document that, from
the judge's side of the screen, does not exist.

So the rule is contextual rather than absolute: **an answer may mention its
evidence only when it was given evidence.** `gate_records` enforces that, and
`build_dataset` runs it before the mix.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: Phrases that only make sense to someone who can see the prompt. Kept narrow
#: and literal: a broad pattern here would eat legitimate agronomy ("the
#: reference plot", "source of nitrogen") and quietly shrink the dataset.
_META = re.compile(
    r"\bthe (?:extracts?|passages?|documents?|references?|reference facts?|"
    r"provided (?:text|material|information)|system prompt|context)\b"
    r"|\bno verified reference\b"
    r"|\bnot (?:given|provided) (?:in|by) the\b"
    r"|\bbased on the (?:extracts?|passages?|documents?|references?|context)\b"
    r"|\baccording to the (?:extracts?|passages?|documents?|references?)\b"
    r"|\bthe (?:library|corpus|manuals? provided)\b"
    r"|\bmy (?:training|instructions|knowledge base|reference material)\b"
    r"|\bas an ai\b|\blanguage model\b",
    re.IGNORECASE,
)


def meta_leaks(answer: str) -> list[str]:
    """Phrases in `answer` that refer to the prompt rather than to farming."""
    if not answer:
        return []
    return [m.group(0).strip() for m in _META.finditer(answer)]


def _was_given_evidence(rec: dict) -> bool:
    """True when the row's prompt actually contains numbered extracts.

    `passage_ids` — not `gate_passages`. A derived closed-book row keeps its
    origin passages for the numeric gate but is rendered *without* them, so it
    may not talk about extracts the reader cannot see.
    """
    return bool(rec.get("meta", {}).get("passage_ids"))


def gate_records(records: Iterable[dict]) -> tuple[list[dict], dict]:
    """Split into (kept, stats), dropping closed-book rows that cite the prompt."""
    kept: list[dict] = []
    stats: dict = {"in": 0, "kept": 0, "dropped": 0, "examples": []}
    for rec in records:
        stats["in"] += 1
        leaks = [] if _was_given_evidence(rec) else meta_leaks(rec.get("a") or "")
        if leaks:
            stats["dropped"] += 1
            if len(stats["examples"]) < 15:
                stats["examples"].append({"id": rec.get("id"), "phrases": leaks})
            continue
        kept.append(rec)
        stats["kept"] += 1
    return kept, stats
