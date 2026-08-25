"""Training rows for the twelve topics must agree with the shipped reference block.

The judged artifact carries the facts in its chat template and the weights are
trained on the same content. If those two disagree the model is being taught to
contradict its own system prompt, which is worse than doing neither, the
2026-08-13 rehearsal showed the model reading "hermetic bags kill storage
insects" and answering "avoid hermetic bags, they do not kill storage insects".
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from sahel_sage.core.config import repo_root
from sahel_sage.core.reference import load_reference
from sahel_sage.training.metaleak import meta_leaks
from sahel_sage.training.numguard import unsupported_quantities
from sahel_sage.training.reference_qa import (
    ANSWERS_PATH,
    QUESTIONS_PATH,
    build_reference_rows,
)


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return build_reference_rows()


def test_every_shipped_fact_has_questions_and_an_answer() -> None:
    """A fact in the block with no training rows is one the weights never learn."""
    facts = {f.id for f in load_reference()}
    questions = json.loads((repo_root() / QUESTIONS_PATH).read_text())["questions"]
    answers = json.loads((repo_root() / ANSWERS_PATH).read_text())["answers"]
    assert facts <= set(questions), f"no questions for: {sorted(facts - set(questions))}"
    assert facts <= set(answers), f"no answer for: {sorted(facts - set(answers))}"


def test_every_topic_is_covered_evenly(rows: list[dict]) -> None:
    counts = Counter(r["meta"]["topic"] for r in rows)
    assert set(counts) == {f.id for f in load_reference()}
    assert len(set(counts.values())) == 1, f"uneven exposure: {counts}"


def test_questions_do_not_collapse(rows: list[dict]) -> None:
    assert len({r["q"].lower() for r in rows}) == len(rows)


def test_no_answer_states_an_untraceable_quantity(rows: list[dict]) -> None:
    """The check that matters: these numbers go into the weights.

    A dose or rate the model learns here is one it will state confidently
    forever, and the inference numeric gate cannot catch it because the model is
    not quoting a retrieved passage, it is recalling.
    """
    sources = {f.id: f.source for f in load_reference()}
    for r in rows:
        bad = unsupported_quantities(r["a"], sources[r["meta"]["topic"]])
        assert not bad, f"{r['meta']['topic']} teaches untraceable {bad}"


def test_no_answer_talks_about_the_prompt(rows: list[dict]) -> None:
    for r in rows:
        assert not meta_leaks(r["a"]), f"{r['id']}: {meta_leaks(r['a'])}"


def test_answers_carry_the_full_contract(rows: list[dict]) -> None:
    for r in rows:
        for section in ("**Likely issue**", "**What to do**", "**Timing**", "**Caution**"):
            assert section in r["a"], f"{r['id']} missing {section}"


def test_rows_claim_no_evidence(rows: list[dict]) -> None:
    """Rendered closed-book, so they must not cite passages the reader cannot see."""
    for r in rows:
        assert r["meta"]["passage_ids"] == []
        assert "[1]" not in r["a"]


@pytest.mark.parametrize(
    ("topic", "must_teach"),
    [
        # Each of these is an error the expert audit found, now taught correctly.
        ("newcastle", "i-2"),                    # not sanitation, not feeding
        ("newcastle", "does not protect a chicken for life"),
        ("ppr", "notifiable"),                   # WOAH-listed, must be reported
        ("faw", "fungicide does nothing"),
        ("salinity", "drain"),                   # drainage before leaching
        ("grain_storage", "hermetic"),
        ("aflatoxin", "never eat, feed or press mouldy groundnuts"),
        ("millet_spacing", "2 to 3 cm"),         # not 100 mm
        ("striga", "before it flowers"),
    ],
)
def test_the_audit_failures_are_taught_correctly(
    rows: list[dict], topic: str, must_teach: str
) -> None:
    answer = next(r["a"] for r in rows if r["meta"]["topic"] == topic)
    assert must_teach.lower() in answer.lower()


def test_every_answer_addresses_the_farmer_in_second_person(rows: list[dict]) -> None:
    """Give the model a "your" to copy, or it will echo the questioner's "my".

    Round 6 answered a Striga question with "Purple flowering weeds, Striga, are
    attached to the roots of my sorghum… My crop is failing". Eight of the
    fifteen topics had no second-person pronoun anywhere in their answer, so the
    only possessive pattern available to copy was the one in the question.
    """
    import re
    by_topic: dict[str, str] = {}
    for r in rows:
        by_topic.setdefault(r["meta"]["topic"], r["a"])
    for topic, answer in by_topic.items():
        assert re.search(r"\b(you|your)\b", answer, re.I), (
            f"{topic} never says 'you' or 'your', the model has no second-person "
            "anchor and will echo the farmer's 'my' back at them"
        )


def test_no_answer_speaks_as_the_farmer(rows: list[dict]) -> None:
    """The advisor never owns the field. "my crop", "my goats" are the reader's."""
    import re
    bad = re.compile(r"\bmy (?:crop|field|sorghum|millet|maize|goats?|sheep|"
                     r"chickens?|animals?|soil|garden|store)\b", re.I)
    for r in rows:
        assert not bad.search(r["a"]), f"{r['id']}: {bad.search(r['a']).group(0)!r}"


def test_generation_is_deterministic() -> None:
    a = build_reference_rows(n_per_topic=10, seed=7)
    b = build_reference_rows(n_per_topic=10, seed=7)
    assert [r["q"] for r in a] == [r["q"] for r in b]
