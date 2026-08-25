"""Corpus-text augmentation for mixed training (round 5).

Round 4 answered domain questions in perfect format with generic content: it
called millet "a cool-season crop" and recommended sanitation instead of
vaccination for Newcastle disease, while answering both correctly when the
manual text was in the prompt. Knowledge comprehensible, not extractable.

The controlled study of exactly this failure (Allen-Zhu & Li, "Physics of
Language Models 3.1", arXiv 2309.14316) measures the cure as a property of the
data, not of the model:

    un-augmented text            9.7%   held-out extraction
    + sentence permutation      70%
    + paraphrase + full names   96.6%
    raw text mixed with Q&A     86.6%

Their finding on our architecture question is equally blunt: on un-augmented
data LoRA scores ~9% *regardless of rank*, and on augmented data it matches
full fine-tuning. So this module, not a bigger model, is the fix.

Three transforms, each cheap and each independently attested:

- `full_names`  , pronouns and bare references replaced by the entity name.
                   FAO prose is pronoun-dense ("it should be sown when…"), and
                   a pronoun teaches nothing retrievable.
- `permute`     , sentence order shuffled within a chunk, so facts are learned
                   as facts rather than as positions in a paragraph.
- `source_tag`  , every example prefixed with its document, which turned a 20x
                   junk-induced capacity loss into 2x in the same series
                   (Physics 3.3).
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from pathlib import Path

from sahel_sage.core.textproc import iter_doc_chunks
from sahel_sage.data.splits import load_holdout

_SENT = re.compile(r"(?<=[.!?])\s+")

#: Pronouns worth resolving. Deliberately conservative: a wrong substitution
#: teaches a wrong fact, which is worse than an unresolved pronoun.
_SUBJECT_PRONOUNS = ("It", "They", "This", "These", "Such plants", "Such animals")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def source_tag(doc_id: str, title: str = "", org: str = "") -> str:
    """`[org · title]` when known, else the document id."""
    if title and org:
        return f"[{org} · {title}]"
    return f"[{doc_id}]"


def full_names(text: str, entity: str) -> str:
    """Replace leading subject pronouns with `entity`.

    Only sentence-initial subject pronouns are touched. Mid-sentence pronouns
    are ambiguous and a wrong resolution injects a false fact into training
    data, the one outcome worse than leaving the pronoun alone.
    """
    if not entity:
        return text
    out = []
    for s in sentences(text):
        for p in _SUBJECT_PRONOUNS:
            if s.startswith(p + " "):
                s = entity + " " + s[len(p) + 1:]
                break
        out.append(s)
    return " ".join(out)


def permute(text: str, rng: random.Random) -> str:
    """Shuffle sentence order. Worth 9.7% -> 70% on its own."""
    parts = sentences(text)
    if len(parts) < 3:
        return text
    rng.shuffle(parts)
    return " ".join(parts)


def entity_of(title: str) -> str:
    """A crude subject for the document, used as the pronoun antecedent.

    Titles look like "Guide to Maize Production" or "Sheep and Goat Handbook";
    the informative noun is usually the first capitalised content word.
    """
    stop = {"A", "An", "The", "Guide", "To", "Manual", "Handbook", "Training",
            "Field", "Farmers", "Farmer's", "Production", "Practices", "For",
            "Of", "And", "On", "In", "Small", "Smallholder"}
    for word in re.findall(r"[A-Za-z][A-Za-z-]+", title):
        if word.capitalize() not in stop and len(word) > 3:
            return word.capitalize()
    return ""


def build_raw_rows(
    corpus_dir: Path,
    sources_path: Path,
    target_words: int = 320,
    min_words: int = 60,
    permutations: int = 1,
    seed: int = 42,
    limit: int | None = None,
) -> Iterator[dict]:
    """Raw corpus chunks as `raw_text` records for mixed training.

    Holdout documents are excluded through `iter_doc_chunks(exclude_docs=…)`,
    the same hook `training.imatrix.build_calibration` uses, and every record
    carries `meta.source_docs` so `mixer.build_dataset`'s holdout assertion
    actually fires on them.
    """
    rng = random.Random(seed)
    holdout = load_holdout()
    meta_by_id = {
        s["id"]: s for s in json.loads(sources_path.read_text())
    }
    emitted = 0
    for chunk_id, text in iter_doc_chunks(
        corpus_dir,
        target_words=target_words,
        overlap_words=0,
        min_words=min_words,
        exclude_docs=holdout,
    ):
        doc_id = chunk_id.split(":")[0]
        src = meta_by_id.get(doc_id, {})
        title, org = src.get("title", ""), src.get("org", "")
        tag = source_tag(doc_id, title, org)
        entity = entity_of(title)

        variants = [full_names(text, entity)]
        for _ in range(max(0, permutations)):
            variants.append(permute(variants[0], rng))

        for v, body in enumerate(variants):
            yield {
                "id": f"{chunk_id}:raw{v}",
                "kind": "raw_text",
                "text": f"{tag}\n{body}",
                "meta": {
                    "source_docs": [doc_id],
                    "passage_ids": [chunk_id],
                    "lang": src.get("lang", "en"),
                    "cluster": src.get("cluster", "?"),
                    "critique": "pass",
                    "augment": "fullname" if v == 0 else "fullname+permute",
                },
            }
            emitted += 1
            if limit and emitted >= limit:
                return


def write_raw_rows(out_path: Path, **kwargs) -> dict:
    n = words = 0
    with out_path.open("w") as f:
        for rec in build_raw_rows(**kwargs):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            words += len(rec["text"].split())
    return {"raw_text_rows": n, "words": words}
