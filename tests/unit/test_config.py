"""configs/*.toml must round-trip through their pydantic models."""

import os
from unittest import mock

import pytest
from pydantic import ValidationError

from sahel_sage.core.config import RetrievalSettings, load_retrieval, load_settings, repo_root


def test_settings_round_trip():
    s = load_settings()
    assert s.bench.threads == 4
    assert s.repo.model_path.endswith(".gguf")
    assert s.lab.path("dir").name == "adtc"


def test_env_override():
    with mock.patch.dict(os.environ, {"SAHEL_BENCH_THREADS": "8"}):
        s = load_settings()
        assert s.bench.threads == 8


def test_repo_root_is_repo():
    assert (repo_root() / "metadata.json").exists()


class TestRetrievalConfig:
    """configs/retrieval.toml existed for two days before any code read it.

    ADR-004 and REPORT.md both described the app as gating on IDF coverage at
    0.72 while it was in fact running the rank heuristic at a hardcoded 0.35.
    These tests exist so that cannot silently recur.
    """

    def test_the_file_is_actually_read(self):
        r = load_retrieval()
        assert r.confidence.threshold == 0.72, (
            "the calibrated threshold (ADR-006, recalibrated for k=8); if this "
            "changed on purpose, update REPORT.md and ADR-006 to match"
        )
        assert r.retrieval.k == 8

    def test_env_override_reaches_the_threshold(self):
        with mock.patch.dict(os.environ, {"SAHEL_CONFIDENCE_THRESHOLD": "0.5"}):
            assert load_retrieval().confidence.threshold == 0.5

    def test_a_missing_threshold_is_a_hard_error(self):
        """No default: falling back to a constant is the bug this file had."""
        with pytest.raises(ValidationError):
            RetrievalSettings.model_validate({"retrieval": {"k": 4}})

    def test_unknown_keys_are_rejected(self):
        with pytest.raises(ValidationError):
            RetrievalSettings.model_validate(
                {"confidence": {"threshold": 0.72, "typo_key": 1}}
            )
