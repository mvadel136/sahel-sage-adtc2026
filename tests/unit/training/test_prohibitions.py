"""Prohibition training rows: the bare model's only defence.

The app screens these questions before the model runs, but the competition's
judged artifact is the bare GGUF and judges chat with it directly. Whatever the
model does unaided is what gets scored, so the refusal has to be in the weights.
"""

from __future__ import annotations

from collections import Counter

import pytest

from sahel_sage.inference.safety import PROHIBITIONS, screen
from sahel_sage.training.prohibitions import (
    PHRASINGS,
    QUALIFIERS,
    build_prohibition_rows,
)


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return build_prohibition_rows(n_per_rule=60, seed=42)


def test_every_prohibition_is_covered(rows: list[dict]) -> None:
    """A rule with no training data is a rule the bare model does not have."""
    covered = {r["meta"]["rule"] for r in rows}
    assert covered == {p.id for p in PROHIBITIONS}


def test_each_rule_gets_its_full_quota(rows: list[dict]) -> None:
    counts = Counter(r["meta"]["rule"] for r in rows)
    assert set(counts.values()) == {60}


def test_questions_do_not_collapse(rows: list[dict]) -> None:
    """`abstain_scope` collapsed from 450 rows to 1 distinct string three times.

    Identical text plus answer-dedup silently emptied the stratum while the row
    count still looked healthy, so distinctness is asserted, not assumed.
    """
    assert len({r["q"].lower() for r in rows}) == len(rows)


def test_every_generated_question_is_still_prohibited(rows: list[dict]) -> None:
    """The generator asserts this internally; this is the regression guard.

    If someone widens PHRASINGS with a question `screen()` does not catch, the
    model would be trained to refuse something the app happily passes through —
    and the two surfaces would disagree about what is dangerous.
    """
    for r in rows:
        assert screen(r["q"]) is not None, r["q"]


def test_the_answer_is_the_shipping_refusal_verbatim(rows: list[dict]) -> None:
    """One source of truth: `a` is `Prohibition.response`, not a paraphrase.

    When these were allowed to drift, the model learned a shortened version that
    dropped the part naming who the farmer should ask instead.
    """
    responses = {p.id: p.response for p in PROHIBITIONS}
    for r in rows:
        assert r["a"] == responses[r["meta"]["rule"]]


def test_qualifiers_actually_appear(rows: list[dict]) -> None:
    """The point of the stratum is that excuses do not work.

    Rows carrying no excuse at all would teach refusal only of the blunt
    phrasing, which is not how anyone asks these questions.
    """
    excuses = [q for q in QUALIFIERS if q]
    with_excuse = sum(any(r["q"].startswith(q) for q in excuses) for r in rows)
    assert with_excuse > len(rows) // 2


def test_rows_carry_no_evidence_and_claim_none(rows: list[dict]) -> None:
    """A refusal is not grounded in a passage, so it must not claim to be.

    Empty `gate_passages` also makes the numeric gate treat any quantity in the
    answer as unsupported — correct here, since these answers must contain none.
    """
    for r in rows:
        assert r["meta"]["passage_ids"] == []
        assert r["meta"]["gate_passages"] == []
        assert "[1]" not in r["a"]


def test_phrasings_cover_every_rule() -> None:
    assert set(PHRASINGS) == {p.id for p in PROHIBITIONS}


def test_generation_is_deterministic() -> None:
    a = build_prohibition_rows(n_per_rule=20, seed=7)
    b = build_prohibition_rows(n_per_rule=20, seed=7)
    assert [r["q"] for r in a] == [r["q"] for r in b]
