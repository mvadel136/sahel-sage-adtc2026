"""Status rows reflect what is on disk; the table renders every column."""

from __future__ import annotations

from sahel_sage.data.sources import Source, SourceRegistry
from sahel_sage.data.status import corpus_status, format_table


def _registry() -> SourceRegistry:
    def src(sid, cluster, lang="en"):
        return Source(
            id=sid, title=sid, org="o", url=f"https://example.org/{sid}",
            topics=[], lang=lang, license_note="", cluster=cluster,
        )

    return SourceRegistry(
        [src("done", "crops"), src("thin", "pest", "fr"), src("raw-only", "sahel"), src("gone", "hort")]
    )


def test_corpus_status_rows(tmp_path):
    raw, txt = tmp_path / "raw", tmp_path / "txt"
    raw.mkdir()
    txt.mkdir()
    (raw / "done.pdf").write_bytes(b"%PDF-")
    (txt / "done.txt").write_text("word " * 600)
    (raw / "thin.html").write_text("<html></html>")
    (txt / "thin.txt").write_text("only a few words here")
    (raw / "raw-only.pdf").write_bytes(b"%PDF-")

    rows = {r["id"]: r for r in corpus_status(_registry(), raw, txt)}
    assert rows["done"] == {
        "id": "done", "cluster": "crops", "lang": "en",
        "raw": True, "txt": True, "words": 600, "status": "extracted",
    }
    assert rows["thin"]["status"] == "low_yield"
    assert rows["thin"]["lang"] == "fr"
    assert rows["raw-only"] == {
        "id": "raw-only", "cluster": "sahel", "lang": "en",
        "raw": True, "txt": False, "words": 0, "status": "fetched",
    }
    assert rows["gone"]["status"] == "missing"


def test_format_table_renders(tmp_path):
    raw, txt = tmp_path / "raw", tmp_path / "txt"
    raw.mkdir()
    txt.mkdir()
    (txt / "done.txt").write_text("word " * 600)
    table = format_table(corpus_status(_registry(), raw, txt))
    lines = table.splitlines()
    assert lines[0].split() == ["id", "cluster", "lang", "raw", "txt", "words", "status"]
    assert set(lines[1]) <= {"-", " "}
    assert len(lines) == 2 + 4
    assert any("extracted" in line for line in lines)
    # aligned columns: every row is the same width as the header rule
    assert len({len(line) for line in lines[1:]}) <= 2


def test_format_table_empty():
    table = format_table([])
    assert "id" in table and "status" in table
