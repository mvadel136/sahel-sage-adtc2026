"""Kallaama corpus extraction — authentic Wolof agricultural language.

Kallaama (Gauthier, Ndiaye & Guissé 2024, Lacuna Fund, CC-BY 4.0) provides
transcribed Senegalese radio/extension speech about agriculture in Wolof,
Pulaar and Sereer, plus written text corpora. We use the TEXT layer only:

- checked Wolof transcriptions (.trs, ~13 h validated): real farmer/extension
  vocabulary and phrasing — the ground truth our NLLB translations and the
  agronomy glossary are anchored to.
- wolof.txt written corpus (~1.1M words, general domain): Wolof LM text.

Attribution requirement (CC-BY): cite the dataset in REPORT.md and the data
card wherever these texts contribute to the model or glossary.

.trs files are Transcriber XML; segments live as text nodes between sync tags.
Code-switched French tokens are marked `word :fra`; we keep the words and drop
the markers (the real language IS code-switched — erasing that would make the
data less authentic, but the markers themselves are transcription metadata).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_TAG_RE = re.compile(r"<[^>]+>")
_FRA_MARK = re.compile(r"\s*:fra\b")
_EVENT_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")  # noise/event annotations
_WS = re.compile(r"\s+")


def clean_segment(text: str) -> str:
    text = _FRA_MARK.sub("", text)
    text = _EVENT_RE.sub(" ", text)
    return _WS.sub(" ", text).strip()


def iter_trs_segments(trs_path: Path, min_words: int = 5) -> list[str]:
    """Extract cleaned speech segments from one Transcriber .trs file."""
    raw = trs_path.read_text(encoding="utf-8", errors="replace")
    out = []
    for piece in _TAG_RE.split(raw):
        seg = clean_segment(piece)
        if len(seg.split()) >= min_words and not seg.startswith("<?"):
            out.append(seg)
    return out


def extract_wolof_segments(
    kallaama_dir: Path,
    out_path: Path,
    checked_only: bool = True,
    min_words: int = 5,
) -> dict:
    """All Wolof transcription segments -> JSONL {text, source_file, checked}."""
    stats = {"files": 0, "segments": 0, "words": 0}
    subsets = ["checked"] if checked_only else ["checked", "raw"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for subset in subsets:
            trs_dir = kallaama_dir / "transcriptions" / subset / "transcriptions-wol" / "trs"
            if not trs_dir.exists():
                continue
            for trs in sorted(trs_dir.glob("*.trs")):
                segs = iter_trs_segments(trs, min_words=min_words)
                stats["files"] += 1
                for seg in segs:
                    f.write(
                        json.dumps(
                            {"text": seg, "source_file": trs.name, "checked": subset == "checked"},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats["segments"] += 1
                    stats["words"] += len(seg.split())
    return stats
