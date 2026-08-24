import sqlite3


def test_build_index_counts(library):
    db_path, stats = library
    assert stats["docs"] == 3
    assert stats["skipped"] == 1  # tiny.txt yields no chunk >= 25 words
    db = sqlite3.connect(str(db_path))
    assert db.execute("SELECT count(*) FROM docs").fetchone()[0] == 3
    assert db.execute("SELECT count(*) FROM chunks").fetchone()[0] == stats["chunks"]
    # the long millet doc must split into several chunks (diversity tests rely on it)
    n_millet = db.execute("SELECT count(*) FROM chunks WHERE doc_id = 'millet_mildew'").fetchone()[
        0
    ]
    assert n_millet >= 3
    # provenance metadata landed
    title, org = db.execute("SELECT title, org FROM docs WHERE id = 'goat_fr'").fetchone()
    assert (title, org) == ("Santé des chèvres", "FAO")


def test_rebuild_deletes_previous_db(library, tmp_path):
    # building twice must not duplicate rows
    import json

    from sahel_sage.retrieval.indexer import build_index

    db_path, stats = library
    txt = db_path.parent / "txt"
    sources = db_path.parent / "corpus_sources.json"
    db2 = tmp_path / "lib.db"
    first = build_index(db2, txt, sources)
    second = build_index(db2, txt, sources)
    assert (
        first
        == second
        == {
            "docs": stats["docs"],
            "chunks": stats["chunks"],
            "skipped": stats["skipped"],
        }
    )
    assert json.loads(sources.read_text())  # fixture intact
