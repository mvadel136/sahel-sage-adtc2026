"""Registry load + validation: bad metadata must fail at load, not mid-pipeline."""

from __future__ import annotations

import json

import pytest

from sahel_sage.data.sources import Source, SourceRegistry

_GOOD = [
    {
        "id": "millet-guide",
        "title": "Millet Production Guide",
        "org": "ICRISAT",
        "url": "https://example.org/millet.pdf",
        "topics": ["millet", "production"],
        "lang": "en",
        "license_note": "public extension document",
        "cluster": "crops",
    },
    {
        "id": "chevres-fr",
        "title": "Santé des chèvres",
        "org": "FAO",
        "url": "https://example.org/chevres.pdf",
        "topics": ["goats", "health"],
        "lang": "fr",
        "license_note": "FAO open publication",
        "cluster": "livestock",
    },
]


def _write(tmp_path, records):
    p = tmp_path / "sources.json"
    p.write_text(json.dumps(records))
    return p


def test_load_and_lookups(tmp_path):
    reg = SourceRegistry.load(_write(tmp_path, _GOOD))
    assert len(reg) == 2
    assert reg.by_id["chevres-fr"].lang == "fr"
    assert reg.by_id["millet-guide"].topics == ["millet", "production"]
    assert [s.id for s in reg.by_cluster("crops")] == ["millet-guide"]
    assert reg.by_cluster("pest") == []
    # provenance defaults are not part of the JSON
    assert reg.by_id["millet-guide"].status == "unknown"
    assert reg.by_id["millet-guide"].sha256 is None


def test_duplicate_id_rejected(tmp_path):
    records = [_GOOD[0], {**_GOOD[1], "id": "millet-guide"}]
    with pytest.raises(ValueError, match="duplicate source id"):
        SourceRegistry.load(_write(tmp_path, records))


def test_bad_cluster_rejected(tmp_path):
    records = [{**_GOOD[0], "cluster": "fisheries"}]
    with pytest.raises(ValueError, match="unknown cluster"):
        SourceRegistry.load(_write(tmp_path, records))


def test_bad_lang_rejected(tmp_path):
    records = [{**_GOOD[0], "lang": "de"}]
    with pytest.raises(ValueError, match="unknown lang"):
        SourceRegistry.load(_write(tmp_path, records))


def test_by_cluster_rejects_unknown_name():
    reg = SourceRegistry([Source(**_GOOD[0])])
    with pytest.raises(ValueError, match="unknown cluster"):
        reg.by_cluster("fish")


def test_real_registry_loads(tmp_path):
    from sahel_sage.core.config import repo_root

    reg = SourceRegistry.load(repo_root() / "training" / "corpus_sources.json")
    assert len(reg) == 56
