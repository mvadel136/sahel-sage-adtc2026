"""Enforce FAO terminology on translated Arabic, because MT gets these wrong.

Translation quality is not the thing that loses the African-language bonus.
Terminology is. NLLB's own model card says it "is not intended to be used with
domain specific texts", and agricultural-veterinary text is exactly that: a
sentence can be fluent, grammatical, natural Arabic and still call Striga by
the name of a different parasitic plant.

The concrete case: `ستريجا` is Striga. `الهالوك` and `جعفيل` are *Orobanche*,
broomrape. Both are real Arabic words for real parasitic weeds of cereals, and
a reviewer checking that the Arabic reads well would pass the substitution
without blinking. An agronomist judge would catch it in one line and stop
trusting everything around it.

So the terms are pinned from FAO AGROVOC (`data/reference/glossary_ar.json`)
and this module checks translations against them, rather than hoping the model
found the right word. Two directions:

* **required**, if the English says "Striga", the Arabic must contain `ستريجا`
* **forbidden**, and it must NOT contain `الهالوك` or `جعفيل`, which name a
  different plant

The forbidden direction is the one that matters. A missing term is a weak
translation; a confidently wrong term is misinformation with a citation-shaped
hole where the authority should be.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sahel_sage.core.config import repo_root

GLOSSARY_PATH = Path("data/reference/glossary_ar.json")

#: Terms that are genuinely different things but sit close enough in meaning
#: that a translator will swap them. Keyed by the English term whose Arabic is
#: at risk; the values are Arabic strings that must NOT appear.
#:
#: Every entry needs a reason, because a wrong entry here silently rejects good
#: translations.
CONFUSABLE: dict[str, tuple[tuple[str, str], ...]] = {
    # Striga hermonthica vs Orobanche spp. Both parasitic weeds of field crops,
    # both called "broomrape-like" in loose usage, entirely different genera and
    # entirely different control.
    "Striga": ((("الهالوك",), "Orobanche (broomrape), a different parasite"),
               (("جعفيل",), "Orobanche (AGROVOC prefLabel)")),
    # Newcastle disease is a poultry paramyxovirus; PPR is a small-ruminant
    # morbillivirus. Our own English model confused these before the reference
    # block existed, so the Arabic must not repeat it.
    "Newcastle disease": ((("طاعون المجترات الصغيرة",), "PPR, a goat/sheep disease"),),
    "peste des petits ruminants": ((("مرض نيوكاسل",), "Newcastle, a poultry disease"),),
}


@dataclass(frozen=True)
class TermCheck:
    """Why a translation was rejected, in terms a reviewer can act on."""

    missing: tuple[str, ...]        # English terms whose Arabic never appeared
    wrong: tuple[tuple[str, str], ...]  # (Arabic string found, what it means)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.wrong


@lru_cache(maxsize=2)
def load_glossary(path: Path | None = None) -> dict[str, dict]:
    p = path or (repo_root() / GLOSSARY_PATH)
    return json.loads(p.read_text())["terms"]


#: Clitics Arabic fuses to the front of a word: the definite article, and the
#: conjunctions/prepositions that stack in front of it.
_PROCLITIC = re.compile(r"\b(?:[وفبكل]?ال|[وفبكل])(?=\S)")

#: Orthographic variants that carry no meaning difference and that a translator
#: chooses freely: alif forms, final ya/alif-maqsura, ta-marbuta.
#: (RUF001 flags Arabic alef and heh as look-alikes for Latin l and o. They are
#: Arabic on purpose, this is the module that folds Arabic orthography.)
_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})  # noqa: RUF001

#: Tashkil (short-vowel marks). Never emitted by us, they cost 5.6x the tokens
#:, but a translator may include them, and they must not defeat a match.
_TASHKIL = re.compile(r"[ً-ْٰـ]")


def normalize_ar(text: str) -> str:
    """Fold Arabic to a form where glossary terms can be found by substring.

    Without this the check rejects correct translations wholesale: AGROVOC gives
    groundnuts as `فول سوداني`, a farmer's sentence says `الفول السوداني`, and
    the bare form is not a substring of the definite one because `ال` attaches
    to *both* words. Arabic fuses its article and several prepositions directly
    onto the stem, so surface matching needs the clitics off first.
    """
    text = _TASHKIL.sub("", text).translate(_FOLD)
    return _PROCLITIC.sub("", text)


def _mentions(english: str, term: str) -> bool:
    """Does the English text use this term? Word-boundary, case-insensitive.

    Multi-word terms are matched loosely on their head noun so that "pearl
    millet" is found in "sow pearl millet", but "millet" alone does not trigger
    the pearl-millet term.
    """
    return re.search(rf"\b{re.escape(term)}\b", english, re.I) is not None


def check(english: str, arabic: str, glossary: dict[str, dict] | None = None) -> TermCheck:
    """Verify an EN->AR translation used the pinned terminology."""
    glossary = glossary if glossary is not None else load_glossary()
    missing: list[str] = []
    wrong: list[tuple[str, str]] = []
    folded = normalize_ar(arabic)

    for term, entry in glossary.items():
        if not _mentions(english, term):
            continue
        ar = entry.get("ar") or ""
        alts = entry.get("ar_alt") or []
        # An AGROVOC altLabel is still correct Arabic for the concept, so any
        # of them satisfies the requirement.
        if ar and not any(
            form and normalize_ar(form) in folded for form in (ar, *alts)
        ):
            missing.append(term)
        for forms, meaning in CONFUSABLE.get(term, ()):
            for form in forms:
                if normalize_ar(form) in folded:
                    wrong.append((form, meaning))

    return TermCheck(missing=tuple(missing), wrong=tuple(wrong))


def report(english: str, arabic: str) -> str:
    """One human-readable line for a rejection log."""
    c = check(english, arabic)
    if c.ok:
        return "terminology ok"
    bits = []
    if c.wrong:
        bits.append("WRONG TERM " + "; ".join(f"{f} = {m}" for f, m in c.wrong))
    if c.missing:
        bits.append("missing " + ", ".join(c.missing))
    return " | ".join(bits)
