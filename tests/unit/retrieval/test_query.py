import sqlite3

import pytest

from sahel_sage.retrieval.query import match_variants, sanitize_fts_query, tokenize
from sahel_sage.retrieval.store import Retriever

NASTY = [
    'millet -"mildew',
    "NEAR(goat, 2)",
    "AND",
    "*:()",
    '"',
    "goat) OR (dog",
    "col:umn value",
    "sorghum* AND NOT millet",
    '"unclosed phrase millet',
    '- - - "" ** millet',
    "chèvre; DROP TABLE docs; --",
]


@pytest.mark.parametrize("q", NASTY)
def test_sanitized_match_never_raises(library, q):
    db_path, _ = library
    db = sqlite3.connect(str(db_path))
    match = sanitize_fts_query(q)
    if match:  # empty means "no query", callers must not MATCH on it
        db.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", (match,)).fetchall()


@pytest.mark.parametrize("q", NASTY)
def test_search_never_raises_on_nasty_input(library, q):
    db_path, _ = library
    r = Retriever(db_path)
    r.search(q)  # must not raise sqlite3.OperationalError


def test_tokenize_drops_stopwords_and_short_tokens():
    assert tokenize("What should I do about my goats?") == ["goats"]


def test_match_variants_are_quoted():
    variants = match_variants("goat diarrhea")
    assert len(variants) == 3
    for v in variants:
        for term in v.replace(" OR ", "\n").replace(" AND ", "\n").splitlines():
            assert term.startswith('"') and term.endswith('"')
    # synonym leg pulls in French manual vocabulary
    assert '"chèvre"' in variants[1]
