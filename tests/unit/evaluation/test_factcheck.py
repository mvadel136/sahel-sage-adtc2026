"""Fact probes must catch the round-4 millet failure."""

from sahel_sage.evaluation.factcheck import PROBES, check

MILLET = next(p for p in PROBES if p.name == "millet_sowing_time")

WRONG = ("**Likely issue**\nMillet is a cool-season crop, so planting is best in the "
         "dry season, which runs from October to May.")
RIGHT = ("**Likely issue**\nMillet is rainfed here.\n\n**What to do**\n"
         "1. Sow as soon as the first good rains have wet the soil.")


def test_catches_the_actual_round4_failure():
    r = check(WRONG, MILLET)
    assert not r["passed"]
    assert r["forbidden_hit"]


def test_accepts_a_correct_answer():
    assert check(RIGHT, MILLET)["passed"]


def test_every_probe_cites_its_corpus_source():
    for p in PROBES:
        assert p.source and len(p.source) > 20, p.name
        assert p.required or p.forbidden, p.name
