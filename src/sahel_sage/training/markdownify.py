"""Convert round-1..3 ALL-CAPS contract answers into the round-4 markdown shape.

ADR-005: human judges chat with the bare model and rank markdown structure
higher than ALL-CAPS labels, which read as machine output. The machine-readable
``STATUS:`` enum is dropped from visible output entirely, the app derives a
more reliable signal from retrieval confidence (ADR-004).

    LIKELY ISSUE: x            ->  **Likely issue**
                                   x
    ACTIONS:\n1. y             ->  **What to do**
                                   1. y
    TIMING: z                  ->  **Timing**
                                   z
    CAUTION: w                 ->  **Caution**
                                   w
    SOURCES: [1]               ->  **Sources** [1]
    SOURCES: NONE              ->  (dropped)
    STATUS: ANSWERED           ->  (dropped)

Deterministic and lossless for the citation markers: ``[n]`` survives verbatim,
because groundedness evaluation counts them. Conversion returns ``None`` when
the input cannot be parsed into at least an issue and one action, the caller
drops the row rather than train on a malformed target.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sahel_sage.core.prompts import (
    SECTION_ACTIONS,
    SECTION_CAUTION,
    SECTION_ISSUE,
    SECTION_SOURCES,
    SECTION_TIMING,
)

_LABEL_RE = re.compile(
    r"^\s*(LIKELY ISSUE|ACTIONS|TIMING|CAUTION|SOURCES|STATUS)\s*:?\s*",
    re.IGNORECASE,
)
_INLINE_LABEL_RE = re.compile(
    r"(?<=\S)[ \t]+(LIKELY ISSUE|ACTIONS|TIMING|CAUTION|SOURCES|STATUS)[ \t]*:"
)
_ACTION_RE = re.compile(r"^\s*(?:\d+[.)]\s*|[-•*]\s*)(.+)$")
_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def split_sections(text: str) -> dict[str, list[str]]:
    """ALL-CAPS labelled text -> {LABEL: [lines]}. Tolerant, never raises."""
    # The 7B teacher occasionally ran the whole contract onto one line. Left
    # inline, the tail (`… TIMING: … STATUS: ANSWERED`) would be swallowed into
    # an action bullet and survive into the markdown target: two such rows
    # were the only ALL-CAPS residue left in dataset-v4.
    text = _INLINE_LABEL_RE.sub(r"\n\1:", text)
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        m = _LABEL_RE.match(line)
        if m:
            current = m.group(1).upper()
            sections.setdefault(current, [])
            rest = line[m.end():].strip()
            if rest:
                sections[current].append(rest)
        elif current is not None:
            sections[current].append(line.rstrip())
    return sections


def render_markdown(
    issue: str,
    actions: list[str],
    timing: str = "",
    caution: str = "",
    sources: list[int] | None = None,
) -> str:
    """The canonical round-4 answer body. One blank line between sections so
    the judge's markdown renderer actually shows them as separate blocks."""
    blocks = [f"{SECTION_ISSUE}\n{issue.strip()}"]
    numbered = "\n".join(f"{i}. {a.strip()}" for i, a in enumerate(actions, 1))
    blocks.append(f"{SECTION_ACTIONS}\n{numbered}")
    if timing.strip():
        blocks.append(f"{SECTION_TIMING}\n{timing.strip()}")
    if caution.strip():
        blocks.append(f"{SECTION_CAUTION}\n{caution.strip()}")
    if sources:
        marks = "".join(f"[{n}]" for n in sorted(set(sources)))
        blocks.append(f"{SECTION_SOURCES} {marks}")
    return "\n\n".join(blocks)


def markdownify(answer: str) -> str | None:
    """Old ALL-CAPS answer -> new markdown answer, or None if unparseable."""
    sections = split_sections(answer)

    def joined(label: str) -> str:
        return " ".join(s.strip() for s in sections.get(label, []) if s.strip()).strip()

    issue = joined("LIKELY ISSUE")

    actions: list[str] = []
    for line in sections.get("ACTIONS", []):
        m = _ACTION_RE.match(line)
        if m and m.group(1).strip():
            actions.append(m.group(1).strip())
        elif line.strip() and not actions:
            actions.append(line.strip())
    actions = actions[:5]

    if not issue or not actions:
        return None

    sources_text = joined("SOURCES")
    sources: list[int] = []
    if "NONE" not in sources_text.upper():
        sources = sorted({int(n) for n in _CITE_RE.findall(sources_text)})

    return render_markdown(
        issue=issue,
        actions=actions,
        timing=joined("TIMING"),
        caution=joined("CAUTION"),
        sources=sources,
    )


def markdownify_pairs(in_path: Path, out_path: Path) -> dict:
    """Rewrite a pairs jsonl in place-of; rows that fail to parse are dropped."""
    stats = {"in": 0, "converted": 0, "dropped": 0}
    with out_path.open("w") as f:
        for line in in_path.read_text().splitlines():
            if not line.strip():
                continue
            stats["in"] += 1
            rec = json.loads(line)
            md = markdownify(rec.get("a", ""))
            if md is None:
                stats["dropped"] += 1
                continue
            rec["a"] = md
            rec.setdefault("meta", {})["format"] = "markdown_v4"
            stats["converted"] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return stats
