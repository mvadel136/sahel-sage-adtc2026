"""THE text-processing module: one garbage filter, one section splitter, one chunker.

Every consumer parameterizes the same implementations:

- retrieval indexing:  section-aware, chunk(target_words=220, overlap_words=40, min_words=25)
- teacher distillation: chunk(target_words=700, overlap_words=80, min_words=120)
- imatrix calibration:  chunk(target_words=350, overlap_words=0,  min_words=50)

Migrated from training/fetch_corpus.py (clean/garbage filter),
app/index_corpus.py (sections + word-tail overlap chunker) and
training/distill.py (chunk ids). The word-tail overlap variant is the single
canonical chunker; the old paragraph-tail variant in distill.py is retired
(no distilled data was ever produced with it, so nothing depends on it).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

VOWELS = set("aeiouyàâäéèêëîïôöùûüœ")

HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?([A-Z][A-Za-z0-9 ,'()/&\-]{6,70})\s*$")

_PAGE_NUM = re.compile(r"[\d ivxlcIVXLC.\-]{1,8}")


def is_wordlike(tok: str) -> bool:
    t = tok.strip(".,;:()[]\"'’-").lower()  # noqa: RUF001: the curly quote is deliberate
    if not t or not t.isalpha():
        return False
    return 1 <= len(t) <= 20 and bool(VOWELS & set(t))


def line_is_text(line: str) -> bool:
    """Reject lines that survived PDF extraction as garbage.

    Some PDFs embed subset fonts with a shifted encoding, so extraction returns
    strings like 'WKHSURGXFWLYLW\\DQG...', syntactically text, semantically
    noise. A line has to look like language to survive.
    """
    toks = line.split()
    if not toks:
        return False
    if max(len(t) for t in toks) > 24:  # word spacing was lost
        return False
    good = sum(1 for t in toks if is_wordlike(t))
    return good / len(toks) >= 0.5


#: Bullet glyphs that PDF extraction leaves stranded mid-sentence, e.g.
#: "A clear discharge from the nose. •• •• Sores in the mouth". They survive
#: chunking in about a tenth of the library, and the model copies them into
#: its answers where a citation belongs.
_STRAY_BULLETS = re.compile(r"(?:[•·▪◦]\s*){1,}")


def strip_bullet_artifacts(text: str) -> str:
    """Remove stranded bullet glyphs from an extracted passage.

    Called on the passage as it leaves the index, so the same cleaned text
    reaches the prompt, the numeric gate and the reader's screen.
    """
    return re.sub(r"[ \t]{2,}", " ", _STRAY_BULLETS.sub("", text)).strip()


def clean_extracted_text(text: str) -> str:
    """Drop page numbers, repeated header/footer noise, and garbage lines."""
    lines = [line.strip() for line in text.splitlines()]
    freq = Counter(line for line in lines if 0 < len(line) < 60)
    noisy = {line for line, c in freq.items() if c >= 5}
    keep = [
        line
        for line in lines
        if line and line not in noisy and not _PAGE_NUM.fullmatch(line) and line_is_text(line)
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(keep))


def split_sections(text: str) -> list[tuple[str, str]]:
    """-> [(section_title, body)]; a document with no headings yields one entry.

    Manuals are written as headed sections; a chunk that keeps its heading
    retrieves far better than a fixed-width window.
    """
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        s = line.strip()
        if s and HEADING.match(s) and len(s.split()) <= 10:
            sections.append((s, []))
        else:
            sections[-1][1].append(line)
    return [(title, "\n".join(body).strip()) for title, body in sections if "\n".join(body).strip()]


def chunk(
    body: str,
    *,
    target_words: int,
    overlap_words: int,
    min_words: int,
) -> list[str]:
    """Paragraph-packing chunker with word-tail overlap between chunks.

    A "paragraph" larger than target_words is hard-split into word windows
    first, extracted manuals often collapse to few blank lines, and without
    this a whole 26k-word document packs into a single chunk (found the hard
    way: `distill --estimate` reported 56 chunks for 56 documents).
    """
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    paras: list[str] = []
    for p in raw_paras:
        words = p.split()
        if len(words) <= target_words:
            paras.append(p)
        else:
            step = max(target_words - overlap_words, 1)
            for start in range(0, len(words), step):
                window = words[start : start + target_words]
                if window:
                    paras.append(" ".join(window))
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for p in paras:
        pw = len(p.split())
        if n + pw > target_words and cur:
            chunks.append("\n\n".join(cur))
            tail = " ".join(" ".join(cur).split()[-overlap_words:]) if overlap_words else ""
            cur = [tail] if tail else []
            n = len(tail.split())
        cur.append(p)
        n += pw
    if cur:
        chunks.append("\n\n".join(cur))
    return [c for c in chunks if len(c.split()) >= min_words]


def chunk_id(doc_stem: str, ordinal: int, text: str) -> str:
    """Stable chunk identifier: '<doc>:<ordinal>:<sha1_8>'."""
    return f"{doc_stem}:{ordinal}:{hashlib.sha1(text.encode()).hexdigest()[:8]}"


def iter_doc_chunks(
    corpus_dir: Path,
    *,
    target_words: int,
    overlap_words: int,
    min_words: int,
    exclude_docs: set[str] | None = None,
) -> Iterator[tuple[str, str]]:
    """-> (chunk_id, text) over every *.txt document, stable across runs.

    `exclude_docs` (doc stems) is the holdout-enforcement hook: pass the frozen
    holdout set and those documents never reach the consumer.
    """
    exclude = exclude_docs or set()
    for doc in sorted(corpus_dir.glob("*.txt")):
        if doc.stem in exclude:
            continue
        text = doc.read_text(errors="replace")
        for ci, c in enumerate(
            chunk(text, target_words=target_words, overlap_words=overlap_words, min_words=min_words)
        ):
            yield chunk_id(doc.stem, ci, c), c
