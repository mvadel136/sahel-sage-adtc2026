from sahel_sage.training.counterprior import CONFLICTS, build_counter_prior


def test_every_conflict_names_the_wrong_belief_and_cites_a_source():
    for c in CONFLICTS:
        assert c.wrong and ("not" in c.wrong.lower()), c.name
        assert len(c.source) > 20, c.name


def test_millet_correction_is_contrastive():
    rows = build_counter_prior(n_per_conflict=3)
    millet = [r for r in rows if r["meta"]["conflict"] == "millet_sowing"]
    assert millet
    a = millet[0]["a"].lower()
    assert "not a cool-season" in a or "not a cool" in a
    assert "onset of the rain" in a


def test_expansion_produces_varied_phrasings():
    rows = build_counter_prior(n_per_conflict=12)
    assert len(rows) == 12 * len(CONFLICTS)
    assert len({r["q"] for r in rows}) >= 15
    assert len({r["a"] for r in rows}) >= 15
