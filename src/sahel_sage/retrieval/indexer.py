"""Build the offline library index from the fetched corpus.

Input: a directory of ``<id>.txt`` files plus ``corpus_sources.json`` for
provenance metadata. Output: a SQLite database (docs + chunks + FTS5).

Chunking is section-aware via the canonical implementation in core.textproc:
manuals are written as headed sections, and a chunk that keeps its heading
retrieves far better than a fixed-width window — the heading is usually the
exact farmer-facing topic ("Controlling storage pests").
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from sahel_sage.core.textproc import chunk, split_sections
from sahel_sage.retrieval.schema import init_db

# Retrieval chunking profile (see core.textproc module docstring).
TARGET_WORDS = 220
OVERLAP_WORDS = 40
MIN_WORDS = 25


def build_index(db_path: Path, txt_dir: Path, sources_json: Path) -> dict:
    """(Re)build the library database; -> {"docs", "chunks", "skipped"}.

    Rebuild semantics: an existing database file is deleted first, so the
    index always reflects exactly the current corpus. Documents that yield no
    usable chunk (too short, all garbage) are skipped entirely rather than
    left as citation-less doc rows.
    """
    sources = (
        {s["id"]: s for s in json.loads(sources_json.read_text())} if sources_json.exists() else {}
    )
    if db_path.exists():
        db_path.unlink()
    db = init_db(sqlite3.connect(str(db_path)))

    n_docs = n_chunks = n_skipped = 0
    rows: list[tuple[str, int, str, str]] = []
    for txt in sorted(txt_dir.glob("*.txt")):
        doc_id = txt.stem
        doc_rows: list[tuple[str, int, str, str]] = []
        ordinal = 0
        for section, body in split_sections(txt.read_text(errors="replace")):
            for c in chunk(
                body,
                target_words=TARGET_WORDS,
                overlap_words=OVERLAP_WORDS,
                min_words=MIN_WORDS,
            ):
                doc_rows.append((doc_id, ordinal, section, c))
                ordinal += 1
        if not doc_rows:
            n_skipped += 1
            continue
        meta = sources.get(doc_id, {})
        db.execute(
            "INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?)",
            (
                doc_id,
                meta.get("title", doc_id),
                meta.get("org", "unknown"),
                meta.get("url", ""),
                meta.get("lang", "en"),
                ", ".join(meta.get("topics", [])),
                meta.get("license_note", ""),
            ),
        )
        rows.extend(doc_rows)
        n_docs += 1
        n_chunks += len(doc_rows)

    db.executemany("INSERT INTO chunks(doc_id, ordinal, section, text) VALUES (?,?,?,?)", rows)
    db.execute(
        "INSERT INTO chunks_fts(rowid, text, section) SELECT rowid, text, section FROM chunks"
    )
    db.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('chunker', ?)",
        (f"section-aware/{TARGET_WORDS}w+{OVERLAP_WORDS}overlap",),
    )
    db.commit()
    db.close()
    return {"docs": n_docs, "chunks": n_chunks, "skipped": n_skipped}
