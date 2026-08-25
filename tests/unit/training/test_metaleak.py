"""An answer must not narrate its own prompt to a farmer.

From the 2026-08-13 judge rehearsal, the model's opening line on a pesticide
container question:

    "The answer is unclear, and no verified reference is given. From what is
     normally recommended: ..."

The reader cannot see any reference, so this is both confusing and a signal that
what follows is a fallback. It is inherited from grounded training rows, where
"the extracts do not cover this" is a perfectly correct thing to say, the gate
therefore has to be contextual, not a blanket ban.
"""

from __future__ import annotations

import pytest

from sahel_sage.training.metaleak import gate_records, meta_leaks

LEAKS = [
    "The answer is unclear, and no verified reference is given. From what is normally recommended: do not.",
    "Based on the extracts, dry the grain to 13% moisture.",
    "The documents do not cover this question.",
    "According to the passages, sow at the onset of the rains.",
    "As an AI language model, I cannot advise on that.",
    "This is not given in the provided information.",
    "My training data does not include that.",
]

#: Real agronomy that a careless pattern would eat. Every one of these is
#: something the model should be free to say.
CLEAN = [
    "Dry the grain to about 13% moisture before storing it in hermetic bags.",
    "Report it to the veterinary services and separate the sick animals.",
    "I don't have reliable information on that. Ask your extension agent.",
    "Use a reference plot to compare the two varieties side by side.",
    "Manure is a good source of nitrogen for millet.",
    "Read the rate from the label on the container.",
    "The source of the water matters: standing water spreads the disease.",
]


@pytest.mark.parametrize("answer", LEAKS)
def test_prompt_commentary_is_caught(answer: str) -> None:
    assert meta_leaks(answer), answer


@pytest.mark.parametrize("answer", CLEAN)
def test_ordinary_advice_is_untouched(answer: str) -> None:
    assert not meta_leaks(answer), f"over-matched: {meta_leaks(answer)}"


def test_grounded_rows_may_discuss_their_extracts() -> None:
    """On the app path the reader *does* see numbered extracts, so saying the
    extracts fall short is honest rather than confusing."""
    grounded = {"id": "g", "a": "The extracts do not cover this.",
                "meta": {"passage_ids": ["c1"]}}
    kept, stats = gate_records([grounded])
    assert [r["id"] for r in kept] == ["g"]
    assert stats["dropped"] == 0


def test_derived_closed_book_rows_may_not() -> None:
    """A row derived from a grounded one keeps `gate_passages` for the numeric
    gate but is rendered without extracts, so it must not mention them."""
    derived = {"id": "d", "a": "The extracts do not cover this.",
               "meta": {"passage_ids": [], "gate_passages": ["c1"]}}
    kept, stats = gate_records([derived])
    assert kept == []
    assert stats["dropped"] == 1
    assert stats["examples"][0]["id"] == "d"


def test_empty_answers_are_not_leaks() -> None:
    assert meta_leaks("") == []
    kept, _ = gate_records([{"id": "x", "a": "", "meta": {}}])
    assert len(kept) == 1
