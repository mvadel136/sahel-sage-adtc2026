"""Fakes for the answer pipeline: no subprocess, no socket, no llama-server.

Everything is handed to the tests as fixtures rather than importable helpers,
because sibling test directories each own a `conftest` module and importing one
by bare name is a collision waiting to happen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sahel_sage.app.context import AppContext
from sahel_sage.retrieval.evidence import Citation
from sahel_sage.retrieval.rank import RRF_K

#: RRF score of a chunk ranked first in one leg, the unit build_pack
#: normalizes against. Two legs agreeing on the same chunk doubles it.
ONE_LEG = 1.0 / (RRF_K + 1)


class FakeBackend:
    """Replays canned generations and records every prompt it was given.

    The recorded prompts are what the repair-budget and train/serve-format
    assertions inspect.
    """

    def __init__(self, *replies: str):
        self.replies = list(replies) or [""]
        self.prompts: list[str] = []

    def complete(self, prompt: str, **opts) -> str:
        self.prompts.append(prompt)
        # The last reply sticks, so a test that only cares about the first call
        # does not have to pad the list.
        return self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]

    def stream(self, prompt: str, **opts):
        text = self.complete(prompt, **opts)
        for i in range(0, len(text), 8):
            yield text[i : i + 8]

    @property
    def calls(self) -> int:
        return len(self.prompts)


class FakeRetriever:
    def __init__(self, citations: list[Citation] | None = None, coverage: float | None = None):
        self.citations = citations or []
        self.searches: list[tuple[str, str, int]] = []
        #: IDF coverage is what the app actually gates on, so tests set it
        #: directly rather than trying to reach a threshold through RRF scores.
        #: Default: full coverage when anything was found, none when not.
        self.coverage = coverage if coverage is not None else (1.0 if self.citations else 0.0)

    def search(self, question: str, lang: str = "en", k: int = 4) -> list[Citation]:
        self.searches.append((question, lang, k))
        return self.citations[:k]

    def coverage_for(self, question: str, citations: list[Citation]) -> float:
        return self.coverage if citations else 0.0

    def stats(self) -> dict:
        return {"documents": 1, "chunks": len(self.citations)}

    def list_documents(self) -> list[dict]:
        return [{"id": "millet_mildew", "title": "Millet Diseases", "org": "ICRISAT"}]


def _citation(n: int, score: float) -> Citation:
    return Citation(
        doc_id=f"doc{n}",
        title=f"Manual {n}",
        org="FAO",
        section=f"Section {n}",
        text=f"Passage number {n} about millet downy mildew control.",
        score=score,
    )


@pytest.fixture
def fake_backend():
    """-> the FakeBackend class; call it with the replies the model should give."""
    return FakeBackend


#: The calibrated production threshold (configs/retrieval.toml). Hardcoded here
#: rather than loaded so a test failure means "behaviour changed", not "someone
#: edited a config".
THRESHOLD = 0.72


@pytest.fixture
def make_ctx():
    def _make(
        backend,
        citations: list[Citation] | None = None,
        coverage: float | None = None,
        **kw,
    ) -> AppContext:
        return AppContext(
            cfg=None,
            backend=backend,
            retriever=FakeRetriever(citations, coverage),
            model_path=Path("model/fake.gguf"),
            threads=4,
            strict_offline=kw.pop("strict_offline", False),
            confidence_threshold=kw.pop("confidence_threshold", THRESHOLD),
            **kw,
        )

    return _make


@pytest.fixture
def strong_citations() -> list[Citation]:
    """Four hits, each confirmed by two retrieval legs. Coverage defaults to 1.0."""
    return [_citation(i, ONE_LEG * 2) for i in range(1, 5)]


@pytest.fixture
def weak_citations() -> list[Citation]:
    """A single lukewarm hit; pair with `coverage=` below THRESHOLD to gate it."""
    return [_citation(1, ONE_LEG)]
