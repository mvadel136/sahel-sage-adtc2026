"""Corpus download: browser-UA urllib with a curl fallback.

Ported from training/fetch_corpus.py. The download mechanics are unchanged
(they were tuned against real institutional hosts); what changed is how the
saved file gets its extension. The legacy script inferred .pdf/.html from the
URL suffix *before* downloading, so six PDFs served from suffix-less URLs
(repository "download" endpoints) were saved as .html and later mis-extracted.
Here the file is downloaded first and named by its magic bytes.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from sahel_sage.data.extract import sniff_format
from sahel_sage.data.sources import Source

# Several institutional hosts reject a non-browser User-Agent outright (403),
# so the request has to look like a browser. curl is the fallback because it
# also handles the TLS/redirect quirks a few of these repositories have.
UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en,fr;q=0.8",
}

# Anything smaller is a soft-404 or interstitial page, not a real document.
MIN_BYTES = 10_000


def fetch(url: str, dest: Path) -> bool:
    """Download ``url`` to ``dest``; True on success (>10KB on disk)."""
    if dest.exists() and dest.stat().st_size > MIN_BYTES:
        return True
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
            f.write(r.read())
        if dest.stat().st_size > MIN_BYTES:
            return True
    except Exception as e:
        print(f"  urllib failed ({e}); trying curl", file=sys.stderr)
    if not shutil.which("curl"):
        return False
    rc = subprocess.run(
        ["curl", "-sL", "--fail", "--max-time", "300", "-A", UA["User-Agent"],
         "-o", str(dest), url],
    ).returncode
    if rc != 0 or not dest.exists() or dest.stat().st_size <= MIN_BYTES:
        print(f"  FETCH FAIL {url}", file=sys.stderr)
        return False
    return True


def fetch_source(source: Source, raw_dir: Path) -> Path | None:
    """Download one source into ``raw_dir``; -> saved path, or None on failure.

    A pre-existing ``<id>.pdf`` or ``<id>.html`` above the size floor is
    reused as-is (re-running the pipeline must be cheap and offline-friendly).
    A fresh download lands under a neutral ``.download`` name and is renamed
    by content sniff, so the extension always tells the truth about the bytes.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".html"):
        existing = raw_dir / f"{source.id}{suffix}"
        if existing.exists() and existing.stat().st_size > MIN_BYTES:
            return existing
    tmp = raw_dir / f"{source.id}.download"
    if not fetch(source.url, tmp):
        tmp.unlink(missing_ok=True)
        source.status = "fetch_failed"
        return None
    final = raw_dir / (source.id + (".pdf" if sniff_format(tmp) == "pdf" else ".html"))
    tmp.replace(final)
    source.status = "fetched"
    return final
