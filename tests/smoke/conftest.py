"""A whole console, wired to a real library index and a fake model.

The point of these tests is the HTTP contract the UI depends on, event names,
their order, and the JSON shapes, so everything below the API is real except
the model itself. No subprocess, no socket, no llama-server.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sahel_sage.retrieval.indexer import build_index

pytest.importorskip("fastapi", reason="the optional 'app' extra is not installed")

from fastapi.testclient import TestClient

from sahel_sage.app.api import create_app
from sahel_sage.app.context import AppContext
from sahel_sage.retrieval.store import open_retriever

ANSWER = """LIKELY ISSUE: Downy mildew on the millet seedlings. [1]
ACTIONS:
1. Uproot and burn the infected plants. [1]
TIMING: Within a week of the first pale leaves. [1]
CAUTION: Call the extension agent if half the field is affected.
SOURCES: [1]
STATUS: ANSWERED"""

_MILLET = (
    "Pearl millet is the staple cereal of the Sahel and downy mildew is its most "
    "damaging disease in the rainy season. Infected millet seedlings show pale "
    "chlorotic leaves and twisted green ears. Farmers should uproot and burn "
    "infected millet plants early, choose resistant pearl millet varieties, and "
    "treat the seed before sowing. Rotating millet with cowpea reduces the downy "
    "mildew inoculum surviving in the soil between two seasons. Demonstration "
    "field number {i} confirmed these downy mildew control practices again."
)


class FakeBackend:
    """Streams a canned contract-shaped answer, eight characters at a time."""

    def __init__(self, text: str = ANSWER):
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt: str, **opts) -> str:
        self.prompts.append(prompt)
        return self.text

    def stream(self, prompt: str, **opts):
        self.prompts.append(prompt)
        for i in range(0, len(self.text), 8):
            yield self.text[i : i + 8]


@pytest.fixture(scope="session")
def library_db(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("library")
    txt = root / "txt"
    txt.mkdir()
    (txt / "millet_mildew.txt").write_text(
        "Millet downy mildew control\n\n" + "\n\n".join(_MILLET.format(i=i) for i in range(12))
    )
    sources = [{"id": "millet_mildew", "title": "Millet Diseases", "org": "ICRISAT", "lang": "en"}]
    (root / "sources.json").write_text(json.dumps(sources))
    db = root / "library.db"
    build_index(db, txt, root / "sources.json")
    return db


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def client(library_db, backend):
    ctx = AppContext(
        cfg=None,
        backend=backend,
        retriever=open_retriever(library_db),
        model_path=library_db.parent / "SahelSage-0.6B-Q4_K_M.gguf",
        threads=4,
        strict_offline=False,  # no strict mode => the watchdog never probes
        # Deliberately 0.0: the smoke tests exercise the HTTP surface (routes,
        # SSE event order), not the abstention policy, and a tiny fixture
        # library would otherwise be refused before any of it ran. The gate
        # itself is tested in tests/unit/app/test_service.py.
        confidence_threshold=0.0,
    )
    with TestClient(create_app(lambda: ctx)) as c:
        yield c
