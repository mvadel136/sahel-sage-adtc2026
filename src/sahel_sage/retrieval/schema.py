"""SQLite schema for the offline library index.

Same table design as app/rag.py (docs, chunks, external-content FTS5, vectors,
meta), with two FTS5 upgrades for a bilingual French/English corpus:

* ``remove_diacritics 2``, accent-insensitive matching ("ble" finds "blé").
  Variant 2 is the corrected algorithm that also folds diacritics expressed as
  combining codepoints; variant 1 misses those.
* ``prefix='2 3'``, 2- and 3-char prefix indexes make ``term*`` queries cheap,
  a poor man's French stemmer (no porter: it is English-only and mangles
  French morphology).

The ``vectors`` table stays in the schema for a future dense leg, but no code
path writes to it, the embedding leg was optional and unused.
"""

from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id TEXT PRIMARY KEY, title TEXT, org TEXT, url TEXT,
    lang TEXT, topics TEXT, license_note TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES docs(id),
    ordinal INTEGER NOT NULL,
    section TEXT,
    text TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, section, content='chunks', content_rowid='rowid',
    tokenize="unicode61 remove_diacritics 2", prefix='2 3'
);
CREATE TABLE IF NOT EXISTS vectors (
    rowid INTEGER PRIMARY KEY REFERENCES chunks(rowid), vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def init_db(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Create all tables on an open connection (idempotent)."""
    conn.executescript(SCHEMA)
    return conn
