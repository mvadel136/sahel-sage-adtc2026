"""The numeric gate must catch the round-4 pesticide invention."""

from sahel_sage.training.numguard import gate_records, quantities, unsupported_quantities

INVENTED = ("**What to do**\n1. Calculate the required millilitres per litre. "
            "For example, if you need 100 ml of pesticide per litre, apply evenly.")
SOURCED = "**What to do**\n1. Apply 250 kg/ha of the compound fertilizer at sowing."
SOURCE_TEXT = "Apply 250 kg/ha of compound fertilizer at sowing time."


def test_detects_measured_quantities():
    assert quantities("apply 100 ml per litre")
    assert quantities("use 250 kg/ha at sowing")


def test_ignores_time_and_length():
    """Steps, weeks and centimetres are reasoning, not doses."""
    assert quantities("wait 2 weeks then weed again") == []
    assert quantities("space plants 30 cm apart") == []


def test_flags_the_round4_pesticide_invention():
    bad = unsupported_quantities(INVENTED, source="")
    assert bad and any("100" in b for b in bad)


def test_accepts_a_quantity_present_in_the_source():
    assert unsupported_quantities(SOURCED, SOURCE_TEXT) == []


def test_closed_book_answers_may_not_state_doses():
    """No passage_ids means no source, so any dose is unsupported by design."""
    rec = {"id": "cb1", "a": INVENTED, "meta": {"passage_ids": []}}
    kept, stats = gate_records([rec], {})
    assert kept == [] and stats["dropped"] == 1


def test_grounded_answer_with_sourced_number_survives():
    rec = {"id": "g1", "a": SOURCED, "meta": {"passage_ids": ["c1"]}}
    kept, stats = gate_records([rec], {"c1": SOURCE_TEXT})
    assert len(kept) == 1 and stats["dropped"] == 0
