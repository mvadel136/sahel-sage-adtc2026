"""Answer-contract parsing, validation, and repair policy.

Since round 4 (ADR-005) the model is trained to emit markdown headings, not
ALL-CAPS labels:

    **Likely issue**
    ...
    **What to do**
    1. ...
    **Timing**
    ...
    **Caution**
    ...
    **Sources** [1][3]          <- only when extracts were supplied

Two consequences for this module:

1. **`Status` is inferred, never parsed.** The visible `STATUS:` enum is gone —
   it read as machine output to a human judge, and the app derives a better
   signal from retrieval confidence (ADR-004). Uncertainty is now expressed in
   natural language, so we classify that language instead.
2. A 0.6B model will not comply perfectly, so parsing stays tolerant by design:
   headings match case-insensitively at line starts with or without the
   asterisks, missing sections degrade the parse instead of failing it, and the
   caller decides between one repair re-prompt and rendering the raw text with
   an "unstructured" notice. Parsing NEVER raises on model output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sahel_sage.core.prompts import (
    SECTION_ACTIONS,
    SECTION_CAUTION,
    SECTION_ISSUE,
    SECTION_SOURCES,
    SECTION_TIMING,
    Status,
)

#: Heading aliases -> canonical key. "ACTIONS" is kept as a tolerated alias so
#: a model that regresses to the round-3 wording still parses.
_HEADINGS = {
    "likely issue": "ISSUE",
    "what to do": "ACTIONS",
    "actions": "ACTIONS",
    "timing": "TIMING",
    "caution": "CAUTION",
    "sources": "SOURCES",
}

_SECTION_RE = re.compile(
    r"^\s*(?:#{1,4}\s*)?(?:\*\*|__)?\s*"
    r"(likely issue|what to do|actions|timing|caution|sources)"
    r"\s*(?:\*\*|__)?\s*:?\s*",
    re.IGNORECASE,
)
_MIDLINE_HEADING_RE = re.compile(
    r"(?<!\n)[ \t]+((?:\*\*|__)\s*"
    r"(?:likely issue|what to do|actions|timing|caution|sources)"
    r"\s*(?:\*\*|__))",
    re.IGNORECASE,
)
_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_ACTION_LINE_RE = re.compile(r"^\s*(?:\d+[.)]\s*|[-•*]\s*)(.+)$")

#: Natural-language markers of the two non-confident states. Order matters:
#: out-of-scope is checked first, because a scope refusal also mentions what it
#: cannot cover.
_OUT_OF_SCOPE_RE = re.compile(
    r"\b(?:is |are |it'?s )?not (?:a |an |about )?(?:[a-z ,'-]{0,40}\b)?"
    r"(?:farming|agricultur\w*|livestock|rural livelihood)"
    r"|outside (?:what|the scope of what|my)\b"
    r"|not (?:something|a topic) (?:i|this advisor) (?:can |could )?(?:help|cover)",
    re.IGNORECASE,
)
_EVIDENCE_LIMITED_RE = re.compile(
    r"\b(?:do|does|don'?t|doesn'?t) not cover\b"
    r"|\bdon'?t cover\b|\bdoesn'?t cover\b"
    r"|\b(?:do|does|don'?t|doesn'?t) not (?:contain|address|mention|include)\b"
    r"|\bno (?:relevant )?(?:extracts?|passages?|documents?)\b"
    r"|\bnot covered by\b"
    r"|\b(?:lack|lacks|lacking|without) (?:the )?(?:specific )?(?:information|evidence|details)\b"
    r"|\b(?:i )?(?:do not|don'?t) have (?:specific |enough |the )?(?:information|evidence|details)\b"
    r"|\bnot enough (?:information|evidence)\b"
    r"|\bnone of the (?:extracts?|passages?|documents?)\b"
    r"|\bnothing in the (?:extracts?|passages?|documents?)\b"
    r"|\b(?:extracts?|passages?|material)\b[^.\n]{0,40}\b(?:do|does|don'?t|doesn'?t) not\b"
    r"|\b(?:general|common|standard) (?:agricultural )?practice\b"
    r"|\bnormally recommended\b",
    re.IGNORECASE,
)


@dataclass
class AnswerContract:
    likely_issue: str = ""
    actions: list[str] = field(default_factory=list)
    timing: str = ""
    caution: str = ""
    sources: list[int] = field(default_factory=list)
    status: Status | None = None

    @property
    def cited(self) -> bool:
        return bool(self.sources)


@dataclass
class ParseResult:
    contract: AnswerContract
    ok: bool                      # all required sections present
    missing: list[str]            # section headings not found
    invalid_citations: list[int]  # cited [n] absent from the evidence pack
    raw_text: str

    @property
    def needs_repair(self) -> bool:
        return not self.ok


def infer_status(text: str) -> Status:
    """Classify the answer's own language into the internal Status enum.

    Never returns None: an answer that hedges about nothing and refuses nothing
    is, by definition, an answer.
    """
    if _OUT_OF_SCOPE_RE.search(text):
        return Status.OUT_OF_SCOPE
    if _EVIDENCE_LIMITED_RE.search(text):
        return Status.EVIDENCE_LIMITED
    return Status.ANSWERED


def parse(text: str, valid_source_ids: set[int] | None = None) -> ParseResult:
    """Parse model output into an AnswerContract. Never raises.

    valid_source_ids: the [n] numbers actually present in the evidence pack;
    citations outside it are reported (and stripped from .sources) — a model
    citing a passage it was never shown is a groundedness defect, not a
    rendering choice.
    """
    # The model sometimes runs a heading straight on from the previous
    # sentence ("…stemborer damage. **What to do**"), and a heading the line
    # loop cannot see turned a perfectly good striga answer into
    # EVIDENCE_LIMITED — plus one wasted repair decode on a CPU where decodes
    # are the whole latency budget. Give every mid-line heading its newline
    # back before parsing.
    text = _MIDLINE_HEADING_RE.sub(lambda m: "\n" + m.group(1), text)

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            current = _HEADINGS[m.group(1).lower()]
            rest = line[m.end():].strip()
            sections.setdefault(current, [])
            if rest:
                sections[current].append(rest)
        elif current:
            sections[current].append(line.rstrip())

    def joined(key: str) -> str:
        return " ".join(s.strip() for s in sections.get(key, []) if s.strip())

    c = AnswerContract()
    c.likely_issue = joined("ISSUE")
    c.timing = joined("TIMING")
    c.caution = joined("CAUTION")

    for line in sections.get("ACTIONS", []):
        m = _ACTION_LINE_RE.match(line)
        if m and m.group(1).strip():
            c.actions.append(m.group(1).strip())
        elif line.strip() and not c.actions:
            # model wrote prose actions without numbering — keep as one action
            c.actions.append(line.strip())
    c.actions = c.actions[:5]

    sources_text = joined("SOURCES")
    if "NONE" in sources_text.upper():
        c.sources = []
    else:
        c.sources = sorted({int(n) for n in _CITE_RE.findall(sources_text)})
        if not c.sources:
            # fall back to citations used inline anywhere in the answer
            c.sources = sorted({int(n) for n in _CITE_RE.findall(text)})

    c.status = infer_status(text)

    invalid: list[int] = []
    if valid_source_ids is not None:
        invalid = [n for n in c.sources if n not in valid_source_ids]
        c.sources = [n for n in c.sources if n in valid_source_ids]

    required_present = {
        SECTION_ISSUE: bool(c.likely_issue),
        SECTION_ACTIONS: bool(c.actions)
        or c.status in (Status.OUT_OF_SCOPE, Status.EVIDENCE_LIMITED),
        SECTION_CAUTION: bool(c.caution) or c.status is Status.OUT_OF_SCOPE,
    }
    missing = [k for k, present in required_present.items() if not present]
    return ParseResult(
        contract=c,
        ok=not missing and not invalid,
        missing=missing,
        invalid_citations=invalid,
        raw_text=text,
    )


REPAIR_INSTRUCTION = (
    "Rewrite your answer in exactly this markdown format, keeping the same advice:\n"
    f"{SECTION_ISSUE}\n<1-2 sentences>\n"
    f"{SECTION_ACTIONS}\n1. <step>\n"
    f"{SECTION_TIMING}\n<when to act>\n"
    f"{SECTION_CAUTION}\n<safety note>\n"
    f"{SECTION_SOURCES} <the [n] numbers you used, or omit this line if you used none>"
)
