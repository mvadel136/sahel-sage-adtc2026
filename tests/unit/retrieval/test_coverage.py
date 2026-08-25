"""Coverage confidence must actually discriminate, the previous heuristic
returned 1.0 for every query (ADR-004)."""

import sqlite3

import pytest

from sahel_sage.retrieval.coverage import coverage_confidence, document_frequencies, idf
from sahel_sage.retrieval.indexer import build_index


@pytest.fixture
def conn(tmp_path):
    txt = tmp_path / "txt"
    txt.mkdir()
    (txt / "millet-guide.txt").write_text(
        "Downy Mildew In Millet\n\n"
        + "Downy mildew attacks millet seedlings during humid weather. "
        * 20
    )
    (tmp_path / "sources.json").write_text(
        '[{"id": "millet-guide", "title": "Millet Guide", "org": "ICRISAT",'
        ' "url": "", "topics": ["millet"], "lang": "en",'
        ' "license_note": "", "cluster": "crops"}]'
    )
    db = tmp_path / "lib.db"
    build_index(db, txt, tmp_path / "sources.json")
    c = sqlite3.connect(db)
    yield c
    c.close()


def test_unknown_terms_have_zero_document_frequency(conn):
    dfs = document_frequencies(conn, ["millet", "wasabi"])
    assert dfs["millet"] > 0
    assert dfs["wasabi"] == 0


def test_unseen_term_outweighs_common_term():
    n = 10_000
    assert idf(0, n) > idf(5_000, n)


def test_covered_question_scores_above_uncovered(conn):
    passages = ["Downy mildew attacks millet seedlings during humid weather."]
    covered = coverage_confidence(conn, ["millet", "mildew"], passages)
    uncovered = coverage_confidence(conn, ["wasabi", "greenhouse"], passages)
    assert covered > 0.9
    assert uncovered < 0.2
    assert covered - uncovered > 0.5  # real separation, not a constant


def test_empty_inputs_are_not_confident(conn):
    assert coverage_confidence(conn, [], ["text"]) == 0.0
    assert coverage_confidence(conn, ["millet"], []) == 0.0


def test_bullet_artifacts_are_stripped_from_passages():
    """PDF extraction strands bullet glyphs mid-sentence in about a tenth of
    the library ("...from the nose. •• •• Sores in the mouth..."). The model
    copies them into its answers where a citation belongs, so they are removed
    as the passage leaves the index — the same cleaned text then reaches the
    prompt, the numeric gate and the reader."""
    from sahel_sage.core.textproc import strip_bullet_artifacts

    dirty = "A clear discharge from the nose. •• •• Sores in the mouth. •• Low fever."
    clean = strip_bullet_artifacts(dirty)
    assert "•" not in clean
    assert "Sores in the mouth." in clean
    assert "  " not in clean
