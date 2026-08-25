"""The Retriever: lexical BM25 legs fused with RRF over the library database.

Lexical-only by design: the advisory runs beside a small model in 8 GB of RAM,
offline, and SQLite's FTS5 index with BM25 needs no extra weights. The dense
``vectors`` table remains in the schema but has no code path yet.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sahel_sage.retrieval.evidence import Citation
from sahel_sage.retrieval.query import match_variants
from sahel_sage.retrieval.rank import cap_per_doc, pool_size, rrf_fuse


class Retriever:
    """BM25 retrieval with Reciprocal Rank Fusion over the SQLite library."""

    def __init__(self, db_path: Path):
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row

    def _fts(self, match: str, limit: int) -> list[int]:
        """One MATCH query -> rowids best-first (bm25: lower is better).

        Sanitized queries should never raise, but a defensive catch keeps a
        malformed MATCH from ever taking the app down.
        """
        try:
            rows = self.db.execute(
                "SELECT c.rowid AS rid FROM chunks_fts f "
                "JOIN chunks c ON c.rowid = f.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r["rid"] for r in rows]

    def search(self, question: str, lang: str = "en", k: int = 4) -> list[Citation]:
        """-> up to k citations, at most 2 per document.

        ``lang`` is accepted for interface stability but unused: one FTS index
        covers both languages thanks to diacritics folding and the bilingual
        synonym map.
        """
        del lang
        pool = pool_size(k)
        rankings = [r for m in match_variants(question) if (r := self._fts(m, pool))]
        if not rankings:
            return []
        top = cap_per_doc(rrf_fuse(rankings), self._doc_of, k)
        out: list[Citation] = []
        for rid, score in top:
            row = self.db.execute(
                "SELECT c.doc_id, c.section, c.text, d.title, d.org, d.url "
                "FROM chunks c JOIN docs d ON d.id = c.doc_id WHERE c.rowid = ?",
                (rid,),
            ).fetchone()
            if row:
                out.append(
                    Citation(
                        doc_id=row["doc_id"],
                        title=row["title"],
                        org=row["org"],
                        section=row["section"] or "",
                        text=row["text"],
                        score=round(score, 5),
                        url=row["url"] or "",
                    )
                )
        return out

    def _doc_of(self, rid: int) -> str | None:
        row = self.db.execute("SELECT doc_id FROM chunks WHERE rowid = ?", (rid,)).fetchone()
        return row["doc_id"] if row else None

    def coverage_for(self, question: str, citations: list[Citation]) -> float:
        """IDF-weighted share of the question's rare terms found in `citations`."""
        from sahel_sage.retrieval.coverage import coverage_confidence
        from sahel_sage.retrieval.query import tokenize

        return coverage_confidence(
            self.db, tokenize(question), [c.text for c in citations]
        )

    def stats(self) -> dict:
        c = self.db.execute("SELECT count(*) n FROM chunks").fetchone()["n"]
        d = self.db.execute("SELECT count(*) n FROM docs").fetchone()["n"]
        meta = {r["key"]: r["value"] for r in self.db.execute("SELECT * FROM meta")}
        return {"documents": d, "chunks": c, **meta}

    def list_documents(self) -> list[dict]:
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT d.id, d.title, d.org, d.url, d.lang, d.topics, "
                "(SELECT count(*) FROM chunks c WHERE c.doc_id = d.id) AS chunks "
                "FROM docs d ORDER BY d.org, d.title"
            )
        ]


class NullRetriever:
    """Used when no library has been indexed yet: the app still answers, and
    says plainly that it is answering from model knowledge alone."""

    def search(self, question: str, lang: str = "en", k: int = 4) -> list[Citation]:
        return []

    def coverage_for(self, question: str, citations: list[Citation]) -> float:
        return 0.0

    def stats(self) -> dict:
        return {"documents": 0, "chunks": 0}

    def list_documents(self) -> list[dict]:
        return []


def open_retriever(db_path: Path) -> Retriever | NullRetriever:
    """A Retriever if the database exists, else the honest empty fallback."""
    if db_path.exists():
        return Retriever(db_path)
    return NullRetriever()
