"""Normalize teacher answers into the strict markdown contract.

The 7B teacher follows the contract loosely ~26% of the time (sections inline
as prose, missing headings, etc.). Training targets must be uniform, so:

- parse each answer with the tolerant contract parser;
- if the issue and at least one action survive, re-emit the canonical markdown
  rendering (a missing caution gets a *sampled* escalation line — safe
  boilerplate, never invented technical advice);
- otherwise DROP the pair (a malformed target teaches malformed output).

`CAUTION_VARIANTS` replaces round-3's single `GENERIC_CAUTION` string. ADR-005:
9,082 rows ending in the same sentence gave a greedy decoder a repetition hook,
and Qwen3's model card names repetition loops as this family's characteristic
failure. Sampling is seeded, so the dataset stays reproducible.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from sahel_sage.inference.contract import AnswerContract, parse
from sahel_sage.training.markdownify import render_markdown

#: Safe under any answer whatsoever.
CAUTIONS_ANY: tuple[str, ...] = (
    "If the problem continues or worsens, contact your local extension agent or veterinarian.",
    "If more than a third of the field or herd is affected, get an extension agent to look at it in person.",
    "If the cause is still unclear after a week, ask an extension agent or a laboratory to check it.",
    "Start small — try this on one plot or a few animals before doing it everywhere.",
    "If you are unsure about any step, ask your extension agent before going ahead.",
    "Note down what you did and when, so you can tell later what actually worked.",
)

#: Only when the answer actually involves a product, spray or dose.
CAUTIONS_CHEMICAL: tuple[str, ...] = (
    "Do not increase any dose beyond what the product label states; ask an agro-dealer if the label is unclear.",
    "Wear gloves and a mask when mixing or spraying, and wash thoroughly afterwards.",
    "Store any product out of reach of children and away from feed, seed and drinking water.",
    "Check the pre-harvest interval on the label and do not harvest before it has passed.",
    "Never use human medicines on animals, and never reuse an empty pesticide container for food or water.",
)

#: Only when the answer is about animals.
CAUTIONS_ANIMAL: tuple[str, ...] = (
    "Call a veterinarian if the animal stops eating, cannot stand, or does not improve in two days.",
    "Keep treated animals and their milk or meat separate from the rest until the withholding period has passed.",
    "Stop and seek advice if the animals show new symptoms after treatment rather than repeating the dose.",
    "Isolate any sick animal from the rest of the herd until you know what you are dealing with.",
)

CAUTION_VARIANTS: tuple[str, ...] = CAUTIONS_ANY + CAUTIONS_CHEMICAL + CAUTIONS_ANIMAL

#: Back-compat default for callers that want one fixed line.
GENERIC_CAUTION = CAUTION_VARIANTS[0]

_CHEMICAL_WORDS = (
    "spray", "pesticide", "insecticide", "herbicide", "fungicide", "chemical", "label",
    "dose", "dosage", "product", "treatment", "drench", "vaccin", "fertilis", "fertiliz",
)
_ANIMAL_WORDS = (
    "animal", "cattle", "cow", "calf", "goat", "sheep", "lamb", "chicken", "poultry",
    "hen", "herd", "flock", "livestock", "camel", "donkey", "milk", "veterinar",
)


def sample_caution(rng: random.Random, text: str = "", cluster: str = "") -> str:
    """A closing caution that actually fits the answer.

    A uniformly sampled caution produces "wear gloves when spraying" under an
    answer about arranging nest boxes — the kind of detail a human judge reads
    as the model not understanding its own advice. So the chemical- and
    animal-specific lines are only offered when the answer mentions those
    things.
    """
    pool = list(CAUTIONS_ANY)
    low = text.lower()
    if any(w in low for w in _CHEMICAL_WORDS):
        pool += CAUTIONS_CHEMICAL
    if cluster == "livestock" or any(w in low for w in _ANIMAL_WORDS):
        pool += CAUTIONS_ANIMAL
    return rng.choice(pool)


def canonical_answer(c: AnswerContract, caution: str | None = None) -> str:
    return render_markdown(
        issue=c.likely_issue,
        actions=c.actions,
        timing=c.timing,
        caution=c.caution or caution or GENERIC_CAUTION,
        sources=c.sources,
    )


def normalize_pairs(
    in_path: Path, out_path: Path, valid_ids: set[int] | None = None, seed: int = 42
) -> dict:
    rng = random.Random(seed)
    stats = {"in": 0, "kept_as_is": 0, "reformatted": 0, "dropped": 0}
    with out_path.open("w") as f:
        for line in in_path.read_text().splitlines():
            if not line.strip():
                continue
            stats["in"] += 1
            rec = json.loads(line)
            r = parse(rec["a"], valid_source_ids=valid_ids or {1})
            if r.ok:
                stats["kept_as_is"] += 1
            elif r.contract.likely_issue and r.contract.actions:
                rec["a"] = canonical_answer(r.contract, caution=sample_caution(rng))
                rec.setdefault("meta", {})["normalized"] = True
                stats["reformatted"] += 1
            else:
                stats["dropped"] += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return stats
