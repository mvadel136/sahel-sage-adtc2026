"""Offline retrieval layer: SQLite FTS5 index, sanitized querying, RRF ranking.

Ported from app/rag.py and app/index_corpus.py. No embedding model is required:
the advisory must run offline on a small laptop, and an FTS5 index with BM25 is
a few megabytes with no extra weights.
"""

from sahel_sage.retrieval.evidence import Citation, EvidencePack, build_pack
from sahel_sage.retrieval.indexer import build_index
from sahel_sage.retrieval.store import NullRetriever, Retriever, open_retriever

__all__ = [
    "Citation",
    "EvidencePack",
    "NullRetriever",
    "Retriever",
    "build_index",
    "build_pack",
    "open_retriever",
]
