"""Citations and the evidence pack the answer pipeline consumes.

Every chunk keeps its document title, organisation and section, because the
answer must be able to say *where* it came from. The EvidencePack adds a cheap
confidence heuristic so the pipeline can decide whether the library actually
covers the question or the model is about to answer from its own knowledge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from sahel_sage.retrieval.rank import RRF_K

#: Fallback only. The value the app actually runs on is the calibrated one in
#: configs/retrieval.toml, loaded by core.config.load_retrieval() and carried on
#: AppContext.confidence_threshold. This constant applies to the rank-based
#: heuristic below and suits direct callers that cannot compute IDF coverage
#: (unit tests, ad-hoc scripts); it is NOT the app's abstention threshold.
DEFAULT_THRESHOLD = 0.35

#: RRF score of a chunk ranked first in exactly one leg, the natural unit for
#: "how strong is the top hit".
_TOP_RANK_SCORE = 1.0 / (RRF_K + 1)


@dataclass
class Citation:
    doc_id: str
    title: str
    org: str
    section: str
    text: str
    score: float
    url: str = ""

    def label(self, n: int) -> str:
        return f"[{n}] {self.title} ({self.org})" + (f" — {self.section}" if self.section else "")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidencePack:
    items: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    sufficient: bool = False


def build_pack(
    citations: list[Citation],
    threshold: float = DEFAULT_THRESHOLD,
    k: int = 4,
    coverage: float | None = None,
) -> EvidencePack:
    """Wrap search results with a normalized confidence heuristic.

    confidence = min(1, hits/k) * (top_rrf_score / (1/(RRF_K+1))), capped at
    1.0: scaled by how full the result set is and how strong the top hit is
    (a top hit confirmed by several legs scores above one rank-1 unit).
    """
    if not citations:
        return EvidencePack(items=[], confidence=0.0, sufficient=False)
    if coverage is not None:
        # IDF-weighted term coverage (retrieval/coverage.py) is the calibrated
        # signal; the rank-based heuristic below saturates and is kept only as
        # a fallback for callers that cannot compute coverage.
        confidence = coverage
    else:
        fullness = min(1.0, len(citations) / k)
        strength = citations[0].score / _TOP_RANK_SCORE
        confidence = min(1.0, fullness * strength)
    return EvidencePack(
        items=citations,
        confidence=round(confidence, 4),
        sufficient=confidence >= threshold,
    )
