"""Render mixed dataset records into (prompt, completion) training pairs.

ALL rendering happens here, in the repo, testably — the Kaggle trainer stays a
dumb consumer of {"prompt", "completion"} rows and masks loss on the prompt
tokens. Two styles, matching ADR-001's two round-1 candidates:

- "raw":    the Base-model path — core.prompts.render_raw(), i.e. exactly the
            plain-text format the official evaluator would present.
- "chatml": the Instruct path — Qwen ChatML with an EMPTY THINK PREFILL baked
            into the prompt so the completion never contains think tokens.

Round-4 additions (ADR-005) — all of them exist because the judge chats with
the **bare model**, and the round-3 model, having never seen a question outside
the full system-prompt + evidence + cue envelope, drifted off-format on a bare
question and generated invented turns until the token cap:

- ``bare`` records are rendered as the raw question ONLY: no system prompt, no
  ``FARMER'S QUESTION:``, no ``SAHEL SAGE:`` cue.
- ``multi_turn`` records render their history through ``render_raw(history=…)``
  and train ONLY the final assistant turn.
- **System-prompt dropout**: ~20% of the non-grounded domain rows are rendered
  with the cues but without the system prompt. The choice is a hash of the
  record id, so it is deterministic, stable across rebuilds, and independent of
  record order.

Grounded rows are left exactly as they were — that is the application path, and
it is the one path we have measured as working.

Record kinds handled: grounded_chunk (evidence pack = its source chunk as
extract [1]), closed_book / abstain_* / greeting / wolof (no extracts), bare,
multi_turn, replay_arc ("text" passthrough — already in the profiler's
Question:/Answer: shape), replay_chat (user/assistant messages).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sahel_sage.core.prompts import EvidenceItem, render_raw, system_prompt, user_message

_CHATML_SYS = "<|im_start|>system\n{sys}<|im_end|>\n"
_CHATML_USER = "<|im_start|>user\n{user}<|im_end|>\n"
_CHATML_ASSISTANT_OPEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
_IM_END = "<|im_end|>"

#: Kinds whose rendering may drop the system prompt. `grounded_chunk` is
#: excluded on purpose: that is the app path and it must stay byte-identical to
#: what `app.service` sends at inference time.
DROPOUT_KINDS = frozenset({
    "closed_book", "abstain_limited", "abstain_scope", "greeting", "wolof", "safety"
})
DEFAULT_DROPOUT = 0.2

#: Share of rows that keep the FULL fifteen-fact block, so the serve-time
#: prompt length stays in distribution. See facts_for().
_FULL_BLOCK_RATE = 0.25

#: Which reference facts a row's cluster plausibly needs. Deliberately generous
#: — a fact that is present but unused costs a few tokens, while a fact that is
#: missing makes the target answer unsupported by its own prompt, which is worse
#: than the problem being solved.
_CLUSTER_FACTS: dict[str, tuple[str, ...]] = {
    "livestock": ("ppr", "newcastle", "haemonchus", "camel_feed", "seasons"),
    "pest": ("faw", "striga", "grain_storage", "aflatoxin", "seasons"),
    "crops": ("sowing", "millet_spacing", "seasons", "manure", "water"),
    "sahel": ("seasons", "water", "dunes", "manure", "salinity"),
    "hort": ("salinity", "water", "manure", "seasons"),
    "safety": ("grain_storage", "aflatoxin", "seasons"),
}


def facts_for(rec: dict) -> tuple | None:
    """The reference facts this row's question actually needs, or None for all.

    The full block is 1,521 of the 1,892 constant prefix tokens carried by every
    closed-book row, which left the question itself at 1.6% of the input. The
    recall probe showed exactly what a model does with that ratio: it learned
    the 65-character opening every refusal shares and diverged the moment the
    answer had to depend on the question.

    Returning `None` means "the whole block", and that is the deliberate default
    whenever a row has no usable metadata. Guessing a subset for a row we cannot
    classify risks stripping the one fact its answer depends on, which turns a
    grounded target into an unsupported one. A slightly long prompt is a cost;
    an answer the prompt cannot justify is a defect.
    """
    from sahel_sage.core.reference import load_reference

    all_facts = load_reference()
    by_id = {f.id: f for f in all_facts}
    meta = rec.get("meta") or {}

    # At inference the block is ALWAYS all full fact block — the chat template
    # has no idea what was asked. Training every row on a short subset would
    # therefore trade one distribution mismatch for another, which is the exact
    # mistake being fixed here (only 22% of rows matched the judged format).
    #
    # So a deterministic slice of rows keeps the full block. Most rows get the
    # signal-to-noise improvement; enough keep the serve-time length that the
    # inference case stays in distribution. Varying the block is also the point
    # in its own right: a constant block can only be memorised, while a block
    # that changes per row has to be READ, which is the behaviour we need.
    #
    # Hashed on the row id, like drop_system_prompt above, so the choice is
    # stable across rebuilds and independent of shuffle order.
    digest = hashlib.sha1(f"facts:{rec.get('id', '')}".encode()).hexdigest()
    if (int(digest[:8], 16) % 1000) < _FULL_BLOCK_RATE * 1000:
        return None

    # reference_topic rows name their fact exactly — no guessing needed.
    topic = meta.get("topic")
    if topic in by_id:
        return (by_id[topic],)

    wanted = set(_CLUSTER_FACTS.get(meta.get("cluster", ""), ()))

    # Anything the question mentions, on top of its cluster. `expand` carries
    # the farmer/manual synonym table (armyworm, weevil, mould, ...) that this
    # vocabulary already lives in, so this reuses matching rather than
    # reinventing it.
    question = rec.get("q") or ""
    if question:
        from sahel_sage.retrieval.query import expand, tokenize

        terms = set(expand(tokenize(question)))
        for f in all_facts:
            hay = set(tokenize(f"{f.topic} {f.text}"))
            if terms & hay:
                wanted.add(f.id)

    if not wanted:
        return None
    # Keep file order: the block is a list, and a stable order means the model
    # sees one layout rather than a permutation per row.
    return tuple(f for f in all_facts if f.id in wanted)


def _chunk_lookup(chunks_path: Path) -> dict[str, dict]:
    return {
        rec["chunk_id"]: rec
        for rec in (json.loads(line) for line in chunks_path.read_text().splitlines() if line.strip())
    }


def _evidence_for(rec: dict, chunks: dict[str, dict]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for i, pid in enumerate(rec.get("meta", {}).get("passage_ids", []), 1):
        c = chunks.get(pid)
        if c is None:
            raise KeyError(f"passage {pid} not found in chunks export (record {rec.get('id')})")
        items.append(
            EvidenceItem(n=i, title=c["title"], org=c["org"], section="", text=c["text"])
        )
    return items


def drop_system_prompt(rec: dict, rate: float = DEFAULT_DROPOUT) -> bool:
    """Deterministic per-record coin flip on the record id.

    A hash, not an RNG: the decision must not depend on the order records
    happen to be shuffled into, so a rebuild with the same data gives the same
    rendering byte for byte.
    """
    if rate <= 0 or rec.get("kind") not in DROPOUT_KINDS:
        return False
    digest = hashlib.sha1(str(rec.get("id", "")).encode()).hexdigest()
    return (int(digest[:8], 16) % 1000) < rate * 1000


def _raw_prompt(question: str, items: list[EvidenceItem], lang: str,
                drop_system: bool, facts=None) -> str:
    """Render one row, optionally with the system prompt stripped back off.

    The strip is by LENGTH, so `facts` must be the identical subset in both
    calls. Pass it to one and not the other and the slice lands mid-sentence,
    leaving prompt debris that no test looks for — the row still trains, just
    on garbage. That is why the subset is computed once by the caller and
    threaded, rather than recomputed here.
    """
    prompt = render_raw(question, items, lang=lang, facts=facts)
    if drop_system:
        prompt = prompt[len(system_prompt(bool(items), lang, facts)) + 2:]
    return prompt


def render_record(
    rec: dict,
    style: str,
    chunks: dict[str, dict],
    dropout: float = DEFAULT_DROPOUT,
) -> dict | None:
    """-> {"prompt", "completion"} or None for records to skip."""
    kind = rec.get("kind", "")

    if kind == "raw_text":
        # Mixed training (Physics of LMs 3.1: 9.7% -> 86.6% held-out
        # extraction). An empty prompt means the trainer masks nothing, so the
        # whole chunk carries LM loss — plain continuation learning on the
        # manuals, interleaved with the Q&A in the same shuffle.
        return {"prompt": "", "completion": rec["text"].rstrip()}

    if kind == "replay_arc":
        # already raw-format text; keep as pure completion continuation data
        text = rec["text"]
        q, _, a = text.partition("\nAnswer:")
        return {"prompt": q + "\nAnswer:", "completion": a.rstrip() or " (see answer)"}

    if kind == "replay_chat":
        msgs = rec["messages"]
        if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
            return None
        if style == "chatml":
            prompt = _CHATML_USER.format(user=msgs[0]["content"]) + _CHATML_ASSISTANT_OPEN
            return {"prompt": prompt, "completion": msgs[-1]["content"].rstrip() + _IM_END}
        return {
            "prompt": f"Question: {msgs[0]['content']}\nAnswer:",
            "completion": " " + msgs[-1]["content"].rstrip(),
        }

    lang = rec.get("meta", {}).get("lang", "en")

    if kind == "multi_turn":
        turns = rec.get("turns") or []
        if len(turns) < 2 or not turns[-1].get("a"):
            return None
        history = [(t["q"], t["a"]) for t in turns[:-1]]
        last = turns[-1]
        if style == "chatml":
            prompt = "".join(
                _CHATML_USER.format(user=q) + f"<|im_start|>assistant\n{a}{_IM_END}\n"
                for q, a in history
            ) + _CHATML_USER.format(user=last["q"]) + _CHATML_ASSISTANT_OPEN
            return {"prompt": prompt, "completion": last["a"].rstrip() + _IM_END}
        return {
            "prompt": render_raw(last["q"], [], lang=lang, history=history,
                                 facts=facts_for(rec)),
            "completion": last["a"].rstrip(),
        }

    if "q" not in rec or "a" not in rec:
        return None

    if kind == "bare":
        # The judge's message with nothing wrapped around it. If the shipped
        # chat template is ignored by their client, THIS is what the model sees.
        if style == "chatml":
            prompt = _CHATML_USER.format(user=rec["q"]) + _CHATML_ASSISTANT_OPEN
            return {"prompt": prompt, "completion": rec["a"].rstrip() + _IM_END}
        return {"prompt": rec["q"].strip(), "completion": "\n" + rec["a"].rstrip()}

    # abstain_limited deliberately shows MISMATCHED passages: the model must
    # learn to decline to cite evidence that doesn't answer the question.
    items = _evidence_for(rec, chunks) if kind in ("grounded_chunk", "abstain_limited") else []
    drop_system = drop_system_prompt(rec, dropout)
    # Only the closed-book branch carries the block at all, so this is a no-op
    # for grounded rows — but computing it once here is what guarantees the two
    # calls inside _raw_prompt agree.
    facts = None if items else facts_for(rec)

    if style == "raw":
        return {
            "prompt": _raw_prompt(rec["q"], items, lang, drop_system, facts),
            "completion": rec["a"].rstrip(),
        }

    user = user_message(rec["q"], items)
    prompt = "" if drop_system else _CHATML_SYS.format(
        sys=system_prompt(bool(items), lang, facts))
    prompt += _CHATML_USER.format(user=user) + _CHATML_ASSISTANT_OPEN
    return {"prompt": prompt, "completion": rec["a"].rstrip() + _IM_END}


def render_dataset(
    train_jsonl: Path,
    chunks_path: Path,
    out_path: Path,
    style: str,
    dropout: float = DEFAULT_DROPOUT,
) -> dict:
    assert style in ("raw", "chatml")
    chunks = _chunk_lookup(chunks_path)
    n_in = n_out = 0
    with out_path.open("w") as f:
        for line in train_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            n_in += 1
            rec = json.loads(line)
            rendered = render_record(rec, style, chunks, dropout=dropout)
            if rendered is not None:
                # The trainer only reads prompt/completion, but carrying `kind`
                # lets a smoke run sample WITHIN stratum instead of taking a head
                # slice. That matters: a smoke test exists to check whether one
                # small stratum was learned, and a naive slice can cut it to a
                # handful of rows, so the run proves nothing and the GPU hour is
                # wasted. One extra short string per row.
                rendered["kind"] = rec.get("kind", "?")
                f.write(json.dumps(rendered, ensure_ascii=False) + "\n")
                n_out += 1
    return {"style": style, "records_in": n_in, "rendered": n_out}
