"""The markdown answer contract (ADR-005). Status is INFERRED, never parsed."""

from sahel_sage.core.prompts import Status
from sahel_sage.inference.contract import parse

GOOD = """**Likely issue**
Downy mildew on millet seedlings. [1]

**What to do**
1. Remove and burn infected plants. [1]
2. Rotate with cowpea next season. [2]

**Timing**
Act within a week of first symptoms. [1]

**Caution**
If more than half the field is affected, call the extension agent.

**Sources** [1][2]"""


def test_good_answer_parses_clean():
    r = parse(GOOD, valid_source_ids={1, 2, 3})
    assert r.ok and not r.missing and not r.invalid_citations
    c = r.contract
    assert c.status is Status.ANSWERED
    assert c.sources == [1, 2]
    assert len(c.actions) == 2
    assert c.actions[0].startswith("Remove and burn")
    assert c.timing.startswith("Act within a week")
    assert c.caution.startswith("If more than half")


def test_invalid_citation_stripped_and_reported():
    r = parse(GOOD, valid_source_ids={1})  # [2] was never shown to the model
    assert not r.ok
    assert r.invalid_citations == [2]
    assert r.contract.sources == [1]


def test_no_sources_section_means_no_citations():
    text = GOOD.split("\n\n**Sources**")[0].replace(" [1]", "").replace(" [2]", "")
    r = parse(text, valid_source_ids=set())
    assert r.contract.sources == []
    assert r.ok


def test_missing_sections_are_reported():
    text = "**Likely issue**\nThe millet is diseased and spreading fast."
    r = parse(text, valid_source_ids=set())
    assert r.needs_repair
    assert "**What to do**" in r.missing and "**Caution**" in r.missing


def test_unstructured_prose_never_raises():
    r = parse("Just water the plants more, they will be fine, trust me.")
    assert not r.ok
    assert "**Likely issue**" in r.missing


def test_case_insensitive_headings_and_bullets():
    text = (
        "**likely issue**\naphids on cowpea [1]\n"
        "**WHAT TO DO**\n- spray neem extract [1]\n- check again after three days\n"
        "**timing**\nearly morning\n**Caution**\nwear gloves\n**sources** [1]"
    )
    r = parse(text, valid_source_ids={1})
    assert r.ok
    assert r.contract.actions[0].startswith("spray neem")
    assert r.contract.sources == [1]


def test_headings_without_asterisks_still_parse():
    text = GOOD.replace("**", "")
    r = parse(text, valid_source_ids={1, 2})
    assert r.ok and len(r.contract.actions) == 2


def test_inline_citation_fallback_when_sources_line_empty():
    text = GOOD.replace("**Sources** [1][2]", "**Sources**")
    r = parse(text, valid_source_ids={1, 2})
    assert r.contract.sources == [1, 2]


def test_actions_capped_at_five():
    actions = "\n".join(f"{i}. step {i}" for i in range(1, 9))
    text = GOOD.replace(
        "1. Remove and burn infected plants. [1]\n2. Rotate with cowpea next season. [2]", actions
    )
    r = parse(text, valid_source_ids={1, 2})
    assert len(r.contract.actions) == 5


class TestInferredStatus:
    """The visible STATUS: enum is gone; the language is classified instead."""

    def test_confident_answer_is_answered(self):
        assert parse(GOOD).contract.status is Status.ANSWERED

    def test_hedging_about_evidence_is_evidence_limited(self):
        text = GOOD.replace(
            "Downy mildew on millet seedlings. [1]",
            "The extracts I was given do not cover this question. From general practice: "
            "this looks like downy mildew.",
        )
        assert parse(text).contract.status is Status.EVIDENCE_LIMITED

    def test_off_topic_refusal_is_out_of_scope(self):
        text = (
            "**Likely issue**\nFootball results is not a farming, livestock or rural "
            "livelihood question, so it is outside what I cover.\n\n"
            "**Caution**\nAsk me about planting, pests or livestock health instead."
        )
        r = parse(text)
        assert r.contract.status is Status.OUT_OF_SCOPE
        assert r.ok  # a scope refusal needs neither actions nor a timing section

    def test_a_greeting_is_not_a_refusal(self):
        text = "Hello, and welcome. I am Sahel Sage, an offline advisor for farmers."
        assert parse(text).contract.status is Status.ANSWERED


def test_midline_heading_is_recognised():
    """The model sometimes runs a heading on from the previous sentence.

    "…reducing yield. **What to do**" turned a good striga answer into
    EVIDENCE_LIMITED and cost a wasted repair decode. A heading is a heading
    wherever the newline went."""
    t = ("**Likely issue**\nStriga is a parasitic weed. **What to do**\n"
         "1. Rotate with cowpea.\n\n**Timing**\nEarly. **Caution**\nAsk the agent.")
    r = parse(t)
    assert r.ok
    assert r.contract.actions
    assert r.contract.caution
