import random

from sahel_sage.core.prompts import CONTRACT_SECTIONS, Status
from sahel_sage.inference.contract import infer_status, parse
from sahel_sage.training.derive import (
    derive_abstain_limited,
    derive_abstain_scope,
    derive_bare_questions,
    derive_closed_book,
    derive_greetings,
    derive_multi_turn,
)
from sahel_sage.training.normalize import CAUTION_VARIANTS

PAIRS = [
    {"id": f"d{i}:0:x#0", "kind": "grounded_chunk",
     "q": f"My {'goats' if i % 2 else 'millet'} look bad, question {i}. What is wrong?",
     "a": (f"**Likely issue**\nIssue {i} with the crop. [1]\n\n"
           f"**What to do**\n1. Do thing {i}. [1]\n2. Then do the other thing {i}. [1]\n\n"
           f"**Timing**\nWithin a week. [1]\n\n"
           f"**Caution**\nCare {i}.\n\n**Sources** [1]"),
     "meta": {"passage_ids": [f"d{i}:0:x"], "cluster": "crops" if i % 2 else "livestock",
              "lang": "en", "critique": "pass"}}
    for i in range(40)
]


# --------------------------------------------------------------------------
# closed book / bare — the judge path
# --------------------------------------------------------------------------


def test_closed_book_has_no_citations_and_no_sources_section():
    out = derive_closed_book(PAIRS, 5, random.Random(1))
    assert out
    for rec in out:
        assert "[1]" not in rec["a"]
        assert "**Sources**" not in rec["a"]
        assert rec["meta"]["passage_ids"] == []


def test_closed_book_is_confident_not_hedged():
    """Regression: v3 made every evidence-free answer say "the offline library
    does not cover this question", which reads as evasion in the judges'
    sandbox where there is no library at all (ADR-005)."""
    out = derive_closed_book(PAIRS, 20, random.Random(1))
    for rec in out:
        low = rec["a"].lower()
        assert "library" not in low
        assert "do not cover" not in low and "general practice" not in low
        assert infer_status(rec["a"]) is Status.ANSWERED


def test_bare_rows_are_marked_bare_and_uncited():
    out = derive_bare_questions(PAIRS, 5, random.Random(2))
    assert out and all(r["kind"] == "bare" for r in out)
    assert all("[" not in r["a"] for r in out)


# --------------------------------------------------------------------------
# abstention
# --------------------------------------------------------------------------


def test_abstain_limited_says_so_once_then_advises_without_citing():
    out = derive_abstain_limited(PAIRS, 5, random.Random(1))
    assert out
    for rec in out:
        assert rec["meta"]["passage_ids"]           # passages ARE shown at render
        assert "[1]" not in rec["a"]                # but nothing may be cited
        assert "**Sources**" not in rec["a"]
        assert infer_status(rec["a"]) is Status.EVIDENCE_LIMITED
        assert parse(rec["a"]).contract.actions     # advice still follows


def test_abstain_scope_is_out_of_scope_and_short():
    out = derive_abstain_scope(6, random.Random(1))
    assert len(out) == 6
    for rec in out:
        assert infer_status(rec["a"]) is Status.OUT_OF_SCOPE
        assert len(rec["a"].split()) < 80


def test_scope_answers_are_distinct_enough_to_survive_dedup():
    """Regression: v2 shipped 1 of 240 scope examples and v3 shipped 120 of 450
    because the targets collapsed under answer-dedup."""
    out = derive_abstain_scope(450, random.Random(1))
    assert len({r["a"] for r in out}) == 450
    assert len({r["q"].lower() for r in out}) == 450


# --------------------------------------------------------------------------
# greetings
# --------------------------------------------------------------------------


def test_greetings_contain_no_contract_headings():
    """A judge's first message must not be answered with a "**Likely issue**"
    block."""
    out = derive_greetings(120)
    assert len(out) == 120
    for rec in out:
        assert not any(section in rec["a"] for section in CONTRACT_SECTIONS)
        assert 15 <= len(rec["a"].split()) <= 90


def test_greetings_name_the_domain_and_survive_dedup():
    out = derive_greetings(120)
    assert len({r["q"].lower() for r in out}) == 120
    assert len({r["a"] for r in out}) == 120
    assert all("Sahel Sage" in r["a"] for r in out)


# --------------------------------------------------------------------------
# multi-turn
# --------------------------------------------------------------------------


def test_multi_turn_buckets_and_shape():
    out = derive_multi_turn(PAIRS, 40, random.Random(3))
    assert out
    assert {r["meta"]["bucket"] for r in out} == {
        "clarification", "refinement", "drill_down", "topic_switch"
    }
    for rec in out:
        assert rec["kind"] == "multi_turn"
        assert len(rec["turns"]) >= 2
        assert all(t["q"] and t["a"] for t in rec["turns"])


def test_multi_turn_final_answers_are_short_and_not_the_full_structure():
    out = derive_multi_turn(PAIRS, 40, random.Random(3))
    for rec in out:
        last = rec["turns"][-1]["a"]
        assert len(last.split()) <= 120
        assert "**Timing**" not in last and "**Sources**" not in last


def test_multi_turn_history_carries_no_empty_sources_section():
    out = derive_multi_turn(PAIRS, 40, random.Random(3))
    for rec in out:
        for turn in rec["turns"][:-1]:
            assert "**Sources**" not in turn["a"]
            assert "[1]" not in turn["a"]


def test_clarification_asks_exactly_one_question_before_answering():
    out = [r for r in derive_multi_turn(PAIRS, 40, random.Random(3))
           if r["meta"]["bucket"] == "clarification"]
    assert out
    for rec in out:
        first = rec["turns"][0]["a"]
        assert first.count("?") == 1
        assert not any(section in first for section in CONTRACT_SECTIONS)


def test_topic_switch_second_question_is_a_different_question():
    out = [r for r in derive_multi_turn(PAIRS, 40, random.Random(3))
           if r["meta"]["bucket"] == "topic_switch"]
    assert out
    for rec in out:
        assert rec["turns"][0]["q"] not in rec["turns"][1]["q"]


# --------------------------------------------------------------------------
# anti-repetition
# --------------------------------------------------------------------------


def test_caution_variants_are_plentiful_and_distinct():
    """Qwen3 loops on repeated boilerplate under greedy decoding (ADR-005), so
    a single GENERIC_CAUTION across thousands of rows is a defect."""
    assert len(set(CAUTION_VARIANTS)) >= 10


def test_derived_strata_use_many_different_cautions():
    rng = random.Random(7)
    rows = derive_closed_book(PAIRS, 40, rng) + derive_abstain_limited(PAIRS, 40, rng)
    cautions = {parse(r["a"]).contract.caution for r in rows}
    assert len(cautions) >= 10
