"""fetch_source without the network: reuse-if-exists and magic-byte naming."""

from __future__ import annotations

import pytest

from sahel_sage.data import fetch as fx
from sahel_sage.data.sources import Source


def _src() -> Source:
    return Source(
        id="doc",
        title="t",
        org="o",
        url="https://example.org/download?id=42",  # no suffix, like the real offenders
        topics=[],
        lang="en",
        license_note="",
        cluster="pest",
    )


def test_existing_file_is_reused_without_fetching(tmp_path, monkeypatch):
    existing = tmp_path / "doc.html"
    existing.write_bytes(b"x" * (fx.MIN_BYTES + 1))

    def no_network(url, dest):
        raise AssertionError("fetch() must not run when a good file exists")

    monkeypatch.setattr(fx, "fetch", no_network)
    assert fx.fetch_source(_src(), tmp_path) == existing


def test_download_named_by_magic_bytes_not_url(tmp_path, monkeypatch):
    # the URL has no .pdf suffix; the bytes say PDF: the file must land as .pdf
    def fake_fetch(url, dest):
        dest.write_bytes(b"%PDF-1.5\n" + b"x" * (fx.MIN_BYTES + 1))
        return True

    monkeypatch.setattr(fx, "fetch", fake_fetch)
    src = _src()
    result = fx.fetch_source(src, tmp_path)
    assert result == tmp_path / "doc.pdf"
    assert result.exists()
    assert not (tmp_path / "doc.download").exists()
    assert src.status == "fetched"


def test_html_download_named_html(tmp_path, monkeypatch):
    def fake_fetch(url, dest):
        dest.write_bytes(b"<html><body>" + b"x" * (fx.MIN_BYTES + 1) + b"</body></html>")
        return True

    monkeypatch.setattr(fx, "fetch", fake_fetch)
    assert fx.fetch_source(_src(), tmp_path) == tmp_path / "doc.html"


def test_failed_fetch_returns_none_and_cleans_up(tmp_path, monkeypatch):
    def fake_fetch(url, dest):
        dest.write_bytes(b"soft 404")  # a partial file must not linger
        return False

    monkeypatch.setattr(fx, "fetch", fake_fetch)
    src = _src()
    assert fx.fetch_source(src, tmp_path) is None
    assert list(tmp_path.iterdir()) == []
    assert src.status == "fetch_failed"


@pytest.mark.parametrize("size", [0, 100])
def test_small_existing_file_is_not_reused(tmp_path, monkeypatch, size):
    (tmp_path / "doc.pdf").write_bytes(b"x" * size)
    calls = []

    def fake_fetch(url, dest):
        calls.append(url)
        dest.write_bytes(b"%PDF-1.5\n" + b"y" * (fx.MIN_BYTES + 1))
        return True

    monkeypatch.setattr(fx, "fetch", fake_fetch)
    assert fx.fetch_source(_src(), tmp_path) == tmp_path / "doc.pdf"
    assert calls  # the undersized file did not satisfy the skip check
