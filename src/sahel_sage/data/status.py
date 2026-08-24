"""Corpus inventory: which sources are fetched, extracted, and healthy.

Replaces eyeballing corpus_raw/ and corpus_txt/ listings. The future CLI
(`sage data status`) renders :func:`format_table`; other code consumes the
row dicts directly.
"""

from __future__ import annotations

from pathlib import Path

from sahel_sage.data.extract import LOW_YIELD_WORDS
from sahel_sage.data.sources import SourceRegistry

_COLUMNS = ("id", "cluster", "lang", "raw", "txt", "words", "status")


def corpus_status(registry: SourceRegistry, raw_dir: Path, txt_dir: Path) -> list[dict]:
    """-> one row per source: {id, cluster, lang, raw, txt, words, status}.

    Status is derived from what is on disk, not from in-memory provenance, so
    it is truthful across process restarts.
    """
    raw_dir, txt_dir = Path(raw_dir), Path(txt_dir)
    rows = []
    for s in registry.all:
        raw = bool(list(raw_dir.glob(f"{s.id}.*")))
        txt_path = txt_dir / f"{s.id}.txt"
        txt = txt_path.exists()
        words = len(txt_path.read_text(errors="replace").split()) if txt else 0
        if txt:
            status = "low_yield" if words < LOW_YIELD_WORDS else "extracted"
        elif raw:
            status = "fetched"
        else:
            status = "missing"
        rows.append(
            {"id": s.id, "cluster": s.cluster, "lang": s.lang,
             "raw": raw, "txt": txt, "words": words, "status": status}
        )
    return rows


def format_table(rows: list[dict]) -> str:
    """Render status rows as a fixed-width plain-text table (no deps)."""
    cells = [[str(r[c]) for c in _COLUMNS] for r in rows]
    widths = [
        max(len(col), max((len(row[i]) for row in cells), default=0))
        for i, col in enumerate(_COLUMNS)
    ]
    lines = [
        "  ".join(col.ljust(w) for col, w in zip(_COLUMNS, widths, strict=True)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in cells]
    return "\n".join(lines)
