"""Text extraction: PDF via pypdf, HTML via a minimal tag-stripper.

Ported from training/fetch_corpus.py with one behavior change: the extractor
is chosen by sniffing the file content (``%PDF-`` magic), never the file
extension. The legacy script trusted the URL suffix, so six PDFs served from
suffix-less URLs were saved as .html and run through the tag-stripper, which
shredded them. Sniffing the bytes makes the extension irrelevant.

Cleaning is delegated to :func:`sahel_sage.core.textproc.clean_extracted_text`
, there must be exactly one garbage filter in the codebase.
"""

from __future__ import annotations

import re
from pathlib import Path

from sahel_sage.core.textproc import clean_extracted_text
from sahel_sage.data.sources import Source

# Below this many cleaned words the document is essentially empty: usually a
# scanned (image-only) PDF or an HTML error page that slipped past the fetch
# size floor. Flagged, not discarded: a human decides whether to replace it.
LOW_YIELD_WORDS = 500

_PDF_MAGIC = b"%PDF-"


def sniff_format(path: Path) -> str:
    """-> 'pdf' or 'html' by content, ignoring the file extension.

    The PDF spec allows junk bytes before the header (some generators emit a
    BOM or a stray newline), so the magic is searched in the first kilobyte
    rather than matched at offset zero.
    """
    with Path(path).open("rb") as f:
        head = f.read(1024)
    return "pdf" if _PDF_MAGIC in head else "html"


def pdf_to_text(path: Path) -> str:
    from pypdf import PdfReader  # optional dependency (extras: data); import lazily

    reader = PdfReader(str(path))
    pages = []
    for pg in reader.pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:
            # a single corrupt page must not sink the whole document
            pages.append("")
    return "\n\n".join(pages)


def html_to_text(path: Path) -> str:
    html = path.read_text(errors="replace")
    html = re.sub(r"(?is)<(script|style|nav|header|footer).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"[ \t]+", " ", text)


def extract(raw_path: Path) -> str:
    """Extract raw text from a downloaded file, routing by content sniff."""
    if sniff_format(raw_path) == "pdf":
        return pdf_to_text(raw_path)
    return html_to_text(raw_path)


def _find_raw(source: Source, raw_dir: Path) -> Path | None:
    # Extension is unknown a priori (and untrustworthy anyway): take whatever
    # the fetch step left for this id.
    matches = sorted(Path(raw_dir).glob(f"{source.id}.*"))
    return matches[0] if matches else None


def extract_source(source: Source, raw_dir: Path, txt_dir: Path) -> dict:
    """Extract + clean one source to ``txt_dir/<id>.txt``.

    -> {id, status: extracted|failed|low_yield, word_count}. Failures are
    reported, never raised, the corpus pipeline fixes URLs and re-runs, so
    one bad document must not abort the batch. The source's provenance
    fields are updated in place.
    """
    raw = _find_raw(source, raw_dir)
    if raw is None:
        source.status = "failed"
        return {"id": source.id, "status": "failed", "word_count": 0}
    try:
        cleaned = clean_extracted_text(extract(raw))
    except Exception:
        source.status = "failed"
        return {"id": source.id, "status": "failed", "word_count": 0}
    words = len(cleaned.split())
    txt_dir = Path(txt_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)
    (txt_dir / f"{source.id}.txt").write_text(cleaned)
    status = "low_yield" if words < LOW_YIELD_WORDS else "extracted"
    source.status = status
    source.word_count = words
    return {"id": source.id, "status": status, "word_count": words}
