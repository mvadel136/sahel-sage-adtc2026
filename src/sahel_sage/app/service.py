"""The answer pipeline: retrieve → render → generate → parse → repair.

Pure logic on purpose. No FastAPI, no subprocess, no globals — the pipeline
takes an AppContext and a ChatBackend port, so every rule below is testable
against a fake backend in milliseconds.

The rule that matters most is `resolve_status`: **the more conservative of
retrieval confidence and the model's own status wins**. A 0.6B model will
happily write ``STATUS: ANSWERED`` over an evidence pack that matched nothing;
letting that reach a farmer as a confident answer is the failure mode this
whole project exists to avoid. Retrieval can only ever make the displayed
status *more* cautious, never less.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from sahel_sage.app.context import AppContext
from sahel_sage.core.prompts import EvidenceItem, Status, render_raw
from sahel_sage.inference.contract import REPAIR_INSTRUCTION, AnswerContract, ParseResult, parse
from sahel_sage.inference.safety import (
    Refusal,
    RefusalReason,
    screen,
    unsupported_quantities,
)
from sahel_sage.retrieval.evidence import EvidencePack, build_pack

MAX_TOKENS = 512
TEMPERATURE = 0.3

#: Shown when retrieval found nothing above the calibrated threshold. Farmer.Chat
#: (Digital Green, GPT-4 in the loop) declines ~25% of queries and reports that
#: rate as a headline metric; a 0.6B model offline should decline far more. The
#: previous behaviour here was to answer anyway behind a caution banner.
NO_EVIDENCE_TEXT = (
    "I don't have reliable information on that.\n\n"
    "My library is a fixed set of agricultural and veterinary manuals, and "
    "nothing in it covers your question closely enough for me to answer without "
    "guessing. Guessing is how advice gets someone's animals or crop killed, so "
    "I would rather tell you plainly that I don't know.\n\n"
    "**Who can help:** your extension agent, your veterinary auxiliary, or a "
    "neighbouring farmer who has dealt with the same problem.\n\n"
    "If you can describe the crop or animal and exactly what you are seeing, try "
    "again — a more specific question sometimes matches something I do have."
)

#: Shown when the model wrote a number that is in no retrieved passage. The
#: answer is discarded rather than shown with the number stripped out: a dose
#: with its digits removed still reads as permission to apply something.
INVENTED_NUMBER_TEXT = (
    "I'm not going to answer that one.\n\n"
    "I drafted an answer and it contained a quantity that does not appear in any "
    "of my sources — which means I invented it. A made-up rate, dose or "
    "measurement is worse than no answer at all, so I have discarded it.\n\n"
    "The passages I found are listed below; read them directly, and take any "
    "measurement from a product label or from your extension agent rather than "
    "from me."
)

#: Increasing caution. `max` over this ranking implements "conservative wins".
_CAUTION_RANK = {
    Status.ANSWERED: 0,
    Status.EVIDENCE_LIMITED: 1,
    Status.OUT_OF_SCOPE: 2,
}


@dataclass
class AnswerResult:
    pack: EvidencePack
    raw_text: str
    parse: ParseResult
    repaired: bool
    status: Status
    #: Set when the pipeline declined to answer. `raw_text` then holds the
    #: human-written refusal and the model was either never called or its output
    #: was discarded.
    refusal: Refusal | None = None

    @property
    def structured(self) -> bool:
        """False when even the repair failed: the UI must then show the raw
        text with an "unformatted" notice rather than empty sections."""
        return self.parse.ok

    @property
    def declined(self) -> bool:
        return self.refusal is not None


#: Refusals map onto the existing three-value Status so the UI, the contract and
#: the evaluation harness need no new state. A prohibition is OUT_OF_SCOPE
#: because we will not answer it at any confidence; a retrieval gap is
#: EVIDENCE_LIMITED because more documents would change the outcome.
_REFUSAL_STATUS = {
    RefusalReason.PROHIBITED: Status.OUT_OF_SCOPE,
    RefusalReason.OUT_OF_CONTEXT: Status.OUT_OF_SCOPE,
    RefusalReason.OUT_OF_CONTENT: Status.EVIDENCE_LIMITED,
    RefusalReason.OUT_OF_COLLECTION: Status.EVIDENCE_LIMITED,
}


def refusal_result(pack: EvidencePack, refusal: Refusal) -> AnswerResult:
    """Wrap a human-written refusal as a finished answer.

    The ParseResult is marked `ok` deliberately: the text is well-formed by
    construction because a person wrote it, so there is nothing for the repair
    path to fix and nothing for the UI to flag as unstructured.
    """
    status = _REFUSAL_STATUS[refusal.reason]
    return AnswerResult(
        pack=pack,
        raw_text=refusal.text,
        parse=ParseResult(
            contract=AnswerContract(status=status),
            ok=True,
            missing=[],
            invalid_citations=[],
            raw_text=refusal.text,
        ),
        repaired=False,
        status=status,
        refusal=refusal,
    )


def resolve_status(pack: EvidencePack, model_status: Status | None) -> Status:
    """Displayed status = the more cautious of retrieval and the model.

    An unparseable status counts as EVIDENCE_LIMITED: an answer we could not
    even read the status line of is not an answer we should present as
    confident.
    """
    floor = Status.ANSWERED if pack.sufficient else Status.EVIDENCE_LIMITED
    claimed = model_status or Status.EVIDENCE_LIMITED
    return max(floor, claimed, key=lambda s: _CAUTION_RANK[s])


def evidence_items(pack: EvidencePack) -> list[EvidenceItem]:
    """Pack citations → the numbered extracts the prompt registry renders.

    The 1-based position *is* the citation number the model must use, and the
    same numbering drives the UI chips, so it is computed exactly once here.
    """
    return [
        EvidenceItem(n=i, title=c.title, org=c.org, section=c.section, text=c.text)
        for i, c in enumerate(pack.items, start=1)
    ]


def retrieve(question: str, lang: str, k: int, ctx: AppContext) -> EvidencePack:
    """Search, then score sufficiency with the *calibrated* signal.

    Both arguments below were missing until 2026-08-13, so the app scored every
    question with the saturating rank heuristic at the hardcoded 0.35 while
    ADR-004 and REPORT.md described it as running on IDF coverage at 0.72. The
    calibration was real; it simply was not connected to anything.
    """
    if k <= 0:
        return build_pack([], threshold=ctx.confidence_threshold)
    citations = ctx.retriever.search(question, lang=lang, k=k)
    return build_pack(
        citations,
        threshold=ctx.confidence_threshold,
        k=k,
        coverage=ctx.retriever.coverage_for(question, citations),
    )


def _defects(result: ParseResult) -> int:
    return len(result.missing) + len(result.invalid_citations)


def _repair_prompt(base_prompt: str, bad_text: str) -> str:
    """Re-ask on the same prompt, showing the model its own broken answer.

    REPAIR_INSTRUCTION says "keeping the same advice", which only means
    something if the advice is still in the context window.
    """
    return f"{base_prompt}{bad_text.strip()}\n\n{REPAIR_INSTRUCTION}\n\nSAHEL SAGE:\n"


def _source_text(pack: EvidencePack) -> str:
    return " ".join(c.text for c in pack.items)


from functools import lru_cache


@lru_cache(maxsize=1)
def _reference_text() -> str:
    """Every verified fact and its source quote, as one searchable string."""
    from sahel_sage.core.reference import load_reference
    return " ".join(f"{f.text} {f.source}" for f in load_reference())


def _finalize(
    prompt: str,
    text: str,
    pack: EvidencePack,
    ctx: AppContext,
) -> AnswerResult:
    """Parse, then spend at most ONE repair generation on a broken contract.

    One, not a loop: a repair costs a second full decode on a CPU-only laptop,
    and a model that failed the format twice will not find it on a third try.
    If the repair comes back no better than the original, the original is kept
    — a retry must never make the answer worse.
    """
    valid_ids = set(range(1, len(pack.items) + 1))
    first = parse(text, valid_source_ids=valid_ids)
    result, repaired = first, False

    if not first.ok:
        retry_text = ctx.backend.complete(
            _repair_prompt(prompt, text),
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        retry = parse(retry_text, valid_source_ids=valid_ids)
        if retry.ok or _defects(retry) < _defects(first):
            result, repaired = retry, True

    # Post-generation numeric gate. The same rule has filtered training data
    # since dataset-v5; the audit showed the model still inventing a 100 mg/kg
    # paracetamol dose and a 100 ml/L pesticide concentration at inference, so
    # it runs here too. A number absent from every retrieved passage was made up
    # — unless it comes from the verified reference block. Those facts are the
    # one other place a number can legitimately live: every quantity in them is
    # test-enforced to appear in its own cited primary source, and the model was
    # trained on them. Without this, the gate discarded a correct maize-storage
    # answer for saying "13% moisture" — our own verified number — because the
    # retrieved passages happened not to repeat it.
    invented = unsupported_quantities(
        result.raw_text, _source_text(pack) + " " + _reference_text()
    )
    if invented:
        return refusal_result(
            pack,
            Refusal(
                reason=RefusalReason.OUT_OF_CONTENT,
                text=INVENTED_NUMBER_TEXT,
                rule="invented_quantity",
            ),
        )

    return AnswerResult(
        pack=pack,
        raw_text=result.raw_text,
        parse=result,
        repaired=repaired,
        # An answer whose structure we could not read is not an answer we
        # present as confident — since ADR-005 the status is INFERRED from the
        # answer's language, so an unparseable blob would otherwise infer
        # ANSWERED by default.
        status=resolve_status(pack, result.contract.status if result.ok else None),
    )


def gate(question: str, pack: EvidencePack) -> Refusal | None:
    """Decide whether to answer at all, before spending a decode.

    Two reasons to decline, in order of severity:

    1. **An absolute prohibition.** Doses, pre-harvest intervals, veterinary
       drugs, human medicines for animals, container reuse, children spraying.
       Never answerable, so retrieval confidence is irrelevant and is not
       consulted.
    2. **Nothing retrieved above the calibrated threshold.** Previously this
       produced an answer behind a caution banner, which is the failure mode the
       whole project exists to avoid: a 0.6B model asked to answer without
       evidence answers from a pretraining prior it should not be trusted with.
    """
    prohibited = screen(question)
    if prohibited is not None:
        return prohibited
    if not pack.sufficient:
        return Refusal(
            reason=RefusalReason.OUT_OF_CONTENT if pack.items else RefusalReason.OUT_OF_COLLECTION,
            text=NO_EVIDENCE_TEXT,
            rule="insufficient_evidence",
        )
    return None


def answer(question: str, lang: str, k: int, ctx: AppContext) -> AnswerResult:
    """One complete, non-streamed answer. The reference implementation."""
    pack = retrieve(question, lang, k, ctx)
    declined = gate(question, pack)
    if declined is not None:
        return refusal_result(pack, declined)
    prompt = render_raw(question, evidence_items(pack), lang=lang)
    text = ctx.backend.complete(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    return _finalize(prompt, text, pack, ctx)


#: ("pack", EvidencePack) → ("token", str)* → ("result", AnswerResult)
StreamEvent = tuple[str, object]


def answer_stream(question: str, lang: str, k: int, ctx: AppContext) -> Iterator[StreamEvent]:
    """Streaming variant: citations up front, tokens as they decode, contract last.

    The pack is emitted before the first token so the reader can show its
    sources while the model is still thinking — on this hardware that is
    several seconds of otherwise blank screen. Any repair happens after the
    stream ends and is not itself streamed: the UI replaces the raw blob with
    the parsed sections when the final event arrives.
    """
    pack = retrieve(question, lang, k, ctx)
    yield ("pack", pack)

    declined = gate(question, pack)
    if declined is not None:
        # Streamed as one piece so the reader sees the same text either way; the
        # model is never called, so there is nothing to decode token by token.
        yield ("token", declined.text)
        yield ("result", refusal_result(pack, declined))
        return

    prompt = render_raw(question, evidence_items(pack), lang=lang)
    pieces: list[str] = []
    for piece in ctx.backend.stream(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
        pieces.append(piece)
        yield ("token", piece)

    yield ("result", _finalize(prompt, "".join(pieces), pack, ctx))
