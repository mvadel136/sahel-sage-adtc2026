"""Extraction routing must follow the bytes, not the filename.

The legacy pipeline trusted the URL suffix, so PDFs served from suffix-less
URLs were saved as .html and shredded by the tag-stripper — these tests pin
the fix.
"""

from __future__ import annotations

from sahel_sage.data import extract as ex
from sahel_sage.data.sources import Source

_SRC = Source(
    id="doc",
    title="t",
    org="o",
    url="https://example.org/doc",
    topics=[],
    lang="en",
    license_note="",
    cluster="crops",
)

# 40 unique wordy lines, each > 60 chars so the header/footer de-noiser and
# page-number filter in clean_extracted_text leave them alone.
_WORDY = "\n".join(
    f"<p>Farmers across the Sahel rotate millet with cowpea to keep sandy "
    f"soils fertile and reduce striga pressure during season {i}.</p>"
    for i in range(40)
)


def test_pdf_magic_routes_to_pdf_even_with_html_suffix(tmp_path, monkeypatch):
    fake = tmp_path / "doc.html"  # wrong suffix on purpose
    fake.write_bytes(b"%PDF-1.7 fake body\n" + b"x" * 200)
    monkeypatch.setattr(ex, "pdf_to_text", lambda p: "pdf path taken")
    monkeypatch.setattr(ex, "html_to_text", lambda p: "html path taken")
    assert ex.extract(fake) == "pdf path taken"


def test_html_content_routes_to_html_even_with_pdf_suffix(tmp_path, monkeypatch):
    fake = tmp_path / "doc.pdf"  # wrong suffix on purpose
    fake.write_text("<html><body>hello</body></html>")
    monkeypatch.setattr(ex, "pdf_to_text", lambda p: "pdf path taken")
    monkeypatch.setattr(ex, "html_to_text", lambda p: "html path taken")
    assert ex.extract(fake) == "html path taken"


def test_sniff_tolerates_junk_before_pdf_header(tmp_path):
    fake = tmp_path / "doc.bin"
    fake.write_bytes(b"\xef\xbb\xbf\n%PDF-1.4\n" + b"x" * 100)
    assert ex.sniff_format(fake) == "pdf"


def test_extract_source_extracted(tmp_path):
    raw, txt = tmp_path / "raw", tmp_path / "txt"
    raw.mkdir()
    (raw / "doc.html").write_text(f"<html><body>{_WORDY}</body></html>")
    res = ex.extract_source(_SRC, raw, txt)
    assert res["status"] == "extracted"
    assert res["word_count"] >= 500
    assert (txt / "doc.txt").exists()
    assert "cowpea" in (txt / "doc.txt").read_text()


def test_extract_source_low_yield(tmp_path):
    raw, txt = tmp_path / "raw", tmp_path / "txt"
    raw.mkdir()
    (raw / "doc.html").write_text(
        "<html><body><p>Only a few real words about millet farming survive here.</p></body></html>"
    )
    res = ex.extract_source(_SRC, raw, txt)
    assert res["status"] == "low_yield"
    assert 0 < res["word_count"] < 500


def test_extract_source_missing_raw_is_failed(tmp_path):
    raw, txt = tmp_path / "raw", tmp_path / "txt"
    raw.mkdir()
    res = ex.extract_source(_SRC, raw, txt)
    assert res == {"id": "doc", "status": "failed", "word_count": 0}
    assert not (txt / "doc.txt").exists()


def test_extract_source_extractor_crash_is_failed(tmp_path, monkeypatch):
    raw, txt = tmp_path / "raw", tmp_path / "txt"
    raw.mkdir()
    (raw / "doc.html").write_bytes(b"%PDF-1.4 pretend pdf")

    def boom(path):
        raise RuntimeError("corrupt file")

    monkeypatch.setattr(ex, "pdf_to_text", boom)
    res = ex.extract_source(_SRC, raw, txt)
    assert res["status"] == "failed"
