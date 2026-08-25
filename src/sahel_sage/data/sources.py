"""Typed view of training/corpus_sources.json.

The JSON stays the single source of truth (it is hand-curated with verified
URLs and license notes); this module only adds structure and validation on
load so that a typo in a cluster name or a duplicated id fails loudly at
startup instead of silently producing a lopsided corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_CLUSTERS = frozenset({"crops", "livestock", "sahel", "pest", "hort"})
VALID_LANGS = frozenset({"en", "fr"})


@dataclass
class Source:
    """One corpus source record, mirroring a corpus_sources.json entry.

    The provenance fields (status/sha256/word_count) are not in the JSON,
    they are filled in by the fetch/extract pipeline so a registry instance
    can carry the outcome of a run without a parallel bookkeeping structure.
    """

    id: str
    title: str
    org: str
    url: str
    topics: list[str] = field(default_factory=list)
    lang: str = "en"
    license_note: str = ""
    cluster: str = ""
    # provenance (filled in by fetch/extract, never read from the JSON)
    status: str = "unknown"
    sha256: str | None = None
    word_count: int | None = None


class SourceRegistry:
    """Validated collection of :class:`Source` records."""

    def __init__(self, sources: list[Source]) -> None:
        self._validate(sources)
        self.all: list[Source] = list(sources)
        self.by_id: dict[str, Source] = {s.id: s for s in sources}

    @staticmethod
    def _validate(sources: list[Source]) -> None:
        seen: set[str] = set()
        for s in sources:
            if s.id in seen:
                raise ValueError(f"duplicate source id: {s.id!r}")
            seen.add(s.id)
            if s.cluster not in VALID_CLUSTERS:
                raise ValueError(
                    f"source {s.id!r}: unknown cluster {s.cluster!r} "
                    f"(expected one of {sorted(VALID_CLUSTERS)})"
                )
            if s.lang not in VALID_LANGS:
                raise ValueError(
                    f"source {s.id!r}: unknown lang {s.lang!r} (expected one of {sorted(VALID_LANGS)})"
                )

    @classmethod
    def load(cls, path: Path) -> SourceRegistry:
        records = json.loads(Path(path).read_text())
        return cls([Source(**r) for r in records])

    def by_cluster(self, name: str) -> list[Source]:
        if name not in VALID_CLUSTERS:
            raise ValueError(f"unknown cluster {name!r} (expected one of {sorted(VALID_CLUSTERS)})")
        return [s for s in self.all if s.cluster == name]

    def __len__(self) -> int:
        return len(self.all)
