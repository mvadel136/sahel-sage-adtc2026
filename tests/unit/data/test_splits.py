"""Leakage guard against the real frozen holdout file."""

from __future__ import annotations

import json

import pytest

from sahel_sage.data.splits import HoldoutViolation, assert_no_holdout, load_holdout


def test_load_real_holdout():
    ids = load_holdout()
    assert len(ids) == 6
    assert "fao-ppr-field-manual" in ids
    assert "fao-faw-ffs-maize-afrique-fr" in ids


def test_assert_no_holdout_passes_on_clean_ids():
    holdout = load_holdout()
    assert_no_holdout(["mofa-ghana-maize-guide", "farmafrica-maize-manual"], holdout)


def test_assert_no_holdout_raises_and_names_offenders():
    holdout = load_holdout()
    with pytest.raises(HoldoutViolation, match="fao-ppr-field-manual"):
        assert_no_holdout(
            ["mofa-ghana-maize-guide", "fao-ppr-field-manual", "icraf-faidherbia-wasat"],
            holdout,
        )


def test_empty_holdout_file_rejected(tmp_path):
    p = tmp_path / "holdout.json"
    p.write_text(json.dumps({"doc_ids": []}))
    with pytest.raises(ValueError, match="no doc_ids"):
        load_holdout(p)
