"""Reciprocal Rank Fusion and the per-document diversity cap.

RRF needs no score calibration between legs, only ranks, which is exactly
right for fusing BM25 lists produced by different MATCH expressions.
"""

from __future__ import annotations

from collections.abc import Callable

RRF_K = 60
MAX_PER_DOC = 2  # one verbose manual must not crowd out a second opinion


def pool_size(k: int) -> int:
    """How deep each leg should retrieve before fusion."""
    return max(k * 6, 30)


def rrf_fuse(rankings: list[list[int]]) -> list[tuple[int, float]]:
    """Fuse ranked rowid lists; -> [(rowid, score)] best-first.

    score = sum over legs of 1 / (RRF_K + position + 1).
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for pos, rid in enumerate(ranking):
            fused[rid] = fused.get(rid, 0.0) + 1.0 / (RRF_K + pos + 1)
    return sorted(fused.items(), key=lambda t: t[1], reverse=True)


def cap_per_doc(
    scored: list[tuple[int, float]],
    doc_of: Callable[[int], str | None],
    k: int,
) -> list[tuple[int, float]]:
    """Keep at most MAX_PER_DOC results per document, stopping at k results.

    ``doc_of`` maps a rowid to its doc_id (None drops the row).
    """
    per_doc: dict[str, int] = {}
    out: list[tuple[int, float]] = []
    for rid, score in scored:
        doc_id = doc_of(rid)
        if doc_id is None or per_doc.get(doc_id, 0) >= MAX_PER_DOC:
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        out.append((rid, score))
        if len(out) >= k:
            break
    return out
