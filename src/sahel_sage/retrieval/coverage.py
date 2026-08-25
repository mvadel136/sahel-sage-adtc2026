"""Does the library actually cover this question?, IDF-weighted term coverage.

The first confidence heuristic (fullness x RRF strength) saturated at 1.0 for
every query, in-corpus or not: FTS5 returns *something* for any query sharing a
common word, and a top hit confirmed by several ranking legs always exceeded one
rank-1 unit. Measured on 300 labelled questions + 20 unanswerable ones it had
exactly zero discriminative power (Youden's J = 0).

The replacement asks the question directly: **were the rare, meaningful words of
the query actually found in the retrieved passages?**

    coverage = sum(idf(t) for matched t) / sum(idf(t) for all query terms t)

with idf(t) = log(N / (1 + df(t))) and df from SQLite's `fts5vocab` table (the
documented way to read an FTS5 index's document frequencies,
sqlite.org/fts5.html#the_fts5vocab_virtual_table). A term the corpus has never
seen has df = 0, so it carries the maximum weight into the denominator and
nothing into the numerator: asking about wasabi or a market price drags
confidence down exactly as it should, while a common word like "my" or "farm"
barely moves it.

Rare-term dominance is the property we want, it is the same intuition BM25
encodes, applied to answerability rather than ranking.
"""

from __future__ import annotations

import math
import sqlite3

VOCAB_TABLE = "chunks_vocab"


def ensure_vocab(conn: sqlite3.Connection) -> None:
    """Create the fts5vocab view over chunks_fts (no storage cost, idempotent)."""
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {VOCAB_TABLE} "
        "USING fts5vocab(chunks_fts, 'row')"
    )


def document_frequencies(conn: sqlite3.Connection, terms: list[str]) -> dict[str, int]:
    """term -> number of chunks containing it (0 when the corpus lacks it)."""
    if not terms:
        return {}
    ensure_vocab(conn)
    placeholders = ",".join("?" * len(terms))
    rows = conn.execute(
        f"SELECT term, doc FROM {VOCAB_TABLE} WHERE term IN ({placeholders})", terms
    ).fetchall()
    found = {r[0]: r[1] for r in rows}
    return {t: found.get(t, 0) for t in terms}


def idf(df: int, n_chunks: int) -> float:
    """Smoothed IDF; an unseen term (df=0) gets the maximum weight.

    Uses the +1 smoothing of scikit-learn's TfidfVectorizer(smooth_idf=True):
    `log((1+N)/(1+df)) + 1`. The trailing +1 matters here, an unsmoothed IDF
    is exactly 0 for a term present in every chunk, which on a small corpus
    zeroes the whole weight sum and makes a fully covered question score 0.0
    (caught by tests/unit/retrieval/test_coverage.py). Every term must retain
    some weight so that "found" always beats "not found".
    """
    return math.log((1 + max(n_chunks, 1)) / (1 + max(df, 0))) + 1.0


def coverage_confidence(
    conn: sqlite3.Connection,
    terms: list[str],
    passages: list[str],
    n_chunks: int | None = None,
) -> float:
    """Fraction of the query's *information* (IDF mass) present in the passages.

    Returns 0.0 when there is nothing to weigh, a query of only stopwords
    tells us nothing about coverage, and claiming confidence there would be a
    false positive in the direction that matters (answering when we should not).
    """
    if not terms or not passages:
        return 0.0
    if n_chunks is None:
        n_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0] or 2

    dfs = document_frequencies(conn, terms)
    haystack = " ".join(passages).lower()

    total = matched = 0.0
    for t in terms:
        w = idf(dfs.get(t, 0), n_chunks)
        total += w
        if t in haystack:
            matched += w
    return round(matched / total, 4) if total else 0.0
