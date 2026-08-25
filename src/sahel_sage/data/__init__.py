"""Data layer: source registry, corpus fetch/extract, split guards, status.

Ported from training/fetch_corpus.py and training/fetch_replay.py with one
deliberate behavior change: document format is decided by magic-byte sniffing
of the downloaded content, not by the URL suffix. The URL-suffix heuristic
saved six PDFs served from suffix-less URLs as .html and mis-extracted them.
"""

# NOTE: the bare `extract`/`fetch` functions are deliberately not re-exported,
# they would shadow the `sahel_sage.data.extract` / `.fetch` submodules on
# `from sahel_sage.data import ...`. Import them from their modules.
from sahel_sage.data.extract import extract_source, sniff_format
from sahel_sage.data.fetch import fetch_source
from sahel_sage.data.sources import Source, SourceRegistry
from sahel_sage.data.splits import HoldoutViolation, assert_no_holdout, load_holdout
from sahel_sage.data.status import corpus_status, format_table

__all__ = [
    "HoldoutViolation",
    "Source",
    "SourceRegistry",
    "assert_no_holdout",
    "corpus_status",
    "extract_source",
    "fetch_source",
    "format_table",
    "load_holdout",
    "sniff_format",
]
