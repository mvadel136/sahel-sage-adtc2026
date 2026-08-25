"""Corpus augmentation for mixed training (Physics-of-LMs 3.1 transforms)."""

import json
import random

from sahel_sage.training.augment import (
    build_raw_rows,
    entity_of,
    full_names,
    permute,
    source_tag,
)


def test_full_names_resolves_leading_pronouns():
    text = "Millet tolerates drought. It should be sown when the rains start."
    out = full_names(text, "Millet")
    assert "Millet should be sown" in out
    assert " It should" not in out


def test_full_names_leaves_midsentence_pronouns_alone():
    """A wrong resolution injects a false fact — worse than a pronoun."""
    text = "Farmers say it grows well when they plant it early."
    assert full_names(text, "Millet") == text


def test_permute_reorders_but_preserves_content():
    text = "One. Two. Three. Four. Five."
    out = permute(text, random.Random(1))
    assert sorted(out.split(". ")) != []
    assert set(out.replace(".", "").split()) == {"One", "Two", "Three", "Four", "Five"}


def test_permute_leaves_short_text_alone():
    assert permute("Only one sentence here.", random.Random(1)) == "Only one sentence here."


def test_source_tag_prefers_org_and_title():
    assert source_tag("d", "Maize Guide", "IITA") == "[IITA · Maize Guide]"
    assert source_tag("doc-id") == "[doc-id]"


def test_entity_of_skips_boilerplate_title_words():
    assert entity_of("A Guide to Maize Production") == "Maize"
    assert entity_of("Sheep and Goat Handbook") == "Sheep"


def test_raw_rows_carry_holdout_metadata(tmp_path):
    """mixer's holdout guard only inspects meta.source_docs — if raw rows lack
    it, holdout text would enter training silently."""
    txt = tmp_path / "txt"
    txt.mkdir()
    (txt / "demo-doc.txt").write_text(
        "Millet is grown widely. " * 40 + "It needs rain at sowing. " * 20
    )
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps([{
        "id": "demo-doc", "title": "Millet Guide", "org": "ICRISAT", "url": "",
        "topics": [], "lang": "en", "license_note": "", "cluster": "crops",
    }]))
    rows = list(build_raw_rows(corpus_dir=txt, sources_path=sources,
                               target_words=100, min_words=20))
    assert rows
    for r in rows:
        assert r["kind"] == "raw_text"
        assert r["meta"]["source_docs"] == ["demo-doc"]
        assert r["text"].startswith("[ICRISAT · Millet Guide]")
    assert any(r["meta"]["augment"] == "fullname+permute" for r in rows)
