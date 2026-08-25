"""The answer pipeline's rules, against a fake backend."""

from __future__ import annotations

from sahel_sage.app.service import answer, answer_stream, resolve_status
from sahel_sage.core.prompts import Status
from sahel_sage.inference.contract import REPAIR_INSTRUCTION
from sahel_sage.retrieval.evidence import EvidencePack

GOOD = """**Likely issue**
Downy mildew on the millet seedlings. [1]

**What to do**
1. Uproot and burn the infected plants. [1]
2. Sow a resistant variety next season. [2]

**Timing**
Within a week of the first pale leaves. [1]

**Caution**
If half the field is affected, call the extension agent.

**Sources** [1][2]"""

PROSE = "Just water them more and they will be fine, no need to worry about it."


def test_grounded_answer_parses_to_answered(fake_backend, make_ctx, strong_citations):
    backend = fake_backend(GOOD)
    ctx = make_ctx(backend, strong_citations)

    r = answer("my millet has pale twisted leaves", "en", 4, ctx)

    assert r.status is Status.ANSWERED
    assert r.structured and not r.repaired
    assert r.parse.contract.sources == [1, 2]
    assert len(r.parse.contract.actions) == 2
    assert backend.calls == 1


def test_prompt_is_the_raw_contract_rendering(fake_backend, make_ctx, strong_citations):
    """Guards the train/serve boundary: the model is Base-trained on this exact
    rendering, so a chat-template regression here is a silent score loss."""
    backend = fake_backend(GOOD)
    ctx = make_ctx(backend, strong_citations)

    answer("my millet has pale leaves", "fr", 4, ctx)

    prompt = backend.prompts[0]
    assert prompt.endswith("SAHEL SAGE:\n")
    assert "EXTRACTS FROM THE OFFLINE LIBRARY:" in prompt
    assert "[1] Manual 1 — FAO, Section 1" in prompt
    assert "Réponds en français" in prompt  # the requested language reached the prompt


def test_low_coverage_refuses_instead_of_answering(fake_backend, make_ctx, weak_citations):
    """Changed 2026-08-13: this used to answer behind a caution banner.

    The expert audit found the banner did nothing, a 0.6B model asked to answer
    without evidence answers from its pretraining prior, fluently and wrongly.
    Below the calibrated threshold the model is now never called at all.
    """
    backend = fake_backend(GOOD)
    ctx = make_ctx(backend, weak_citations, coverage=0.30)

    r = answer("something the library barely covers", "en", 4, ctx)

    assert not r.pack.sufficient
    assert r.declined
    assert r.refusal.rule == "insufficient_evidence"
    assert r.status is Status.EVIDENCE_LIMITED
    assert backend.calls == 0, "the model must not be consulted without evidence"


def test_empty_library_refuses(fake_backend, make_ctx):
    backend = fake_backend(GOOD)
    ctx = make_ctx(backend, [])

    r = answer("anything at all", "en", 4, ctx)

    assert r.pack.items == []
    assert r.declined
    assert backend.calls == 0


def test_out_of_scope_survives_a_confident_pack(fake_backend, make_ctx, strong_citations):
    """A retrieval hit does not stop the model declaring the question off-topic.

    Uses a strong pack because a weak one would be refused before the model ran,
    which would test the gate rather than the status merge.
    """
    backend = fake_backend("**Likely issue**\nThat is not a farming question, so I cannot help.")
    ctx = make_ctx(backend, strong_citations)

    r = answer("who won the world cup", "en", 4, ctx)

    assert r.status is Status.OUT_OF_SCOPE
    assert backend.calls == 1  # a valid out-of-scope answer needs no repair


def test_malformed_output_triggers_exactly_one_repair(fake_backend, make_ctx, strong_citations):
    backend = fake_backend(PROSE, GOOD)
    ctx = make_ctx(backend, strong_citations)

    r = answer("my millet has pale leaves", "en", 4, ctx)

    assert backend.calls == 2
    assert r.repaired and r.structured
    assert r.status is Status.ANSWERED
    assert REPAIR_INSTRUCTION in backend.prompts[1]
    assert PROSE in backend.prompts[1]  # the model can see the advice it must reformat


def test_two_failures_degrade_gracefully(fake_backend, make_ctx, strong_citations):
    backend = fake_backend(PROSE, "still not the format")
    ctx = make_ctx(backend, strong_citations)

    r = answer("my millet has pale leaves", "en", 4, ctx)

    assert backend.calls == 2  # the repair budget is exactly one
    assert not r.structured
    assert r.raw_text == PROSE  # a worse retry never replaces the original
    assert r.status is Status.EVIDENCE_LIMITED  # an unreadable status is not confident


def test_citations_outside_the_pack_are_stripped(fake_backend, make_ctx, strong_citations):
    """The model cites [7]; only [1]..[4] were ever shown to it."""
    cited_seven = GOOD.replace("**Sources** [1][2]", "**Sources** [1][7]")
    backend = fake_backend(cited_seven, cited_seven)
    ctx = make_ctx(backend, strong_citations)

    r = answer("my millet has pale leaves", "en", 4, ctx)

    assert r.parse.contract.sources == [1]
    assert r.parse.invalid_citations == [7]
    assert not r.structured


def test_k_zero_skips_retrieval_and_therefore_refuses(fake_backend, make_ctx, strong_citations):
    """k=0 asks for a closed-book answer, which is exactly what we no longer do.

    The parameter still short-circuits retrieval, but an empty pack now means a
    refusal rather than an unevidenced answer.
    """
    backend = fake_backend(GOOD)
    ctx = make_ctx(backend, strong_citations)

    r = answer("closed book question", "en", 0, ctx)

    assert ctx.retriever.searches == []
    assert r.pack.items == []
    assert r.declined
    assert backend.calls == 0


def test_stream_yields_pack_then_tokens_then_result(fake_backend, make_ctx, strong_citations):
    backend = fake_backend(GOOD)
    ctx = make_ctx(backend, strong_citations)

    events = list(answer_stream("my millet has pale leaves", "en", 4, ctx))

    kinds = [k for k, _ in events]
    assert kinds[0] == "pack"
    assert kinds[-1] == "result"
    assert set(kinds[1:-1]) == {"token"}
    assert "".join(p for k, p in events if k == "token") == GOOD
    assert events[-1][1].status is Status.ANSWERED


class TestTheGateIsActuallyWired:
    """Both halves of the sufficiency decision must reach `retrieve`.

    The bug this guards against is not a wrong threshold, it is a threshold
    that is computed, documented, and then never passed in. Each test moves one
    input and requires the outcome to follow.
    """

    def test_the_context_threshold_decides(self, fake_backend, make_ctx, strong_citations):
        answered = answer(
            "my millet has pale leaves", "en", 4,
            make_ctx(fake_backend(GOOD), strong_citations, coverage=0.50, confidence_threshold=0.40),
        )
        refused = answer(
            "my millet has pale leaves", "en", 4,
            make_ctx(fake_backend(GOOD), strong_citations, coverage=0.50, confidence_threshold=0.60),
        )

        assert not answered.declined, "coverage above the threshold must answer"
        assert refused.declined, "the same coverage below the threshold must refuse"

    def test_the_coverage_score_decides(self, fake_backend, make_ctx, strong_citations):
        """Not the RRF rank heuristic: the citations are identical in both runs."""
        low = answer(
            "my millet has pale leaves", "en", 4,
            make_ctx(fake_backend(GOOD), strong_citations, coverage=0.10),
        )
        high = answer(
            "my millet has pale leaves", "en", 4,
            make_ctx(fake_backend(GOOD), strong_citations, coverage=0.99),
        )

        assert low.declined and not high.declined
        assert low.pack.confidence == 0.10
        assert high.pack.confidence == 0.99


class TestProhibitionGate:
    """A prohibited question must never reach the model, however good retrieval is."""

    QUESTION = "How many millilitres of pesticide should I mix per litre of water?"

    def test_prohibited_question_never_reaches_the_model(
        self, fake_backend, make_ctx, strong_citations
    ):
        backend = fake_backend(GOOD)
        ctx = make_ctx(backend, strong_citations)

        r = answer(self.QUESTION, "en", 4, ctx)

        assert backend.calls == 0
        assert r.declined and r.refusal.rule == "pesticide_rate"
        assert r.status is Status.OUT_OF_SCOPE
        assert r.structured, "a human-written refusal needs no repair pass"

    def test_refusal_streams_as_a_single_piece(self, fake_backend, make_ctx, strong_citations):
        backend = fake_backend(GOOD)
        ctx = make_ctx(backend, strong_citations)

        events = list(answer_stream(self.QUESTION, "en", 4, ctx))

        assert [k for k, _ in events] == ["pack", "token", "result"]
        assert events[-1][1].declined
        assert backend.calls == 0

    def test_the_refusal_text_is_what_the_reader_gets(
        self, fake_backend, make_ctx, strong_citations
    ):
        ctx = make_ctx(fake_backend(GOOD), strong_citations)

        r = answer(self.QUESTION, "en", 4, ctx)

        assert "label" in r.raw_text.lower()
        assert r.raw_text == r.refusal.text


def test_an_invented_quantity_discards_the_answer(fake_backend, make_ctx, strong_citations):
    """The audit's exact failure: a dose that appears in no retrieved passage.

    The fixture passages talk about millet downy mildew and contain no numbers,
    so the 100 ml/L below is unsupported by construction.
    """
    invented = GOOD.replace(
        "1. Uproot and burn the infected plants. [1]",
        "1. Mix 100 ml per litre of water and spray. [1]",
    )
    backend = fake_backend(invented)
    ctx = make_ctx(backend, strong_citations)

    r = answer("my millet has pale twisted leaves", "en", 4, ctx)

    assert r.declined and r.refusal.rule == "invented_quantity"
    assert "100 ml" not in r.raw_text, "the invented number must not survive in the output"
    assert r.pack.items, "the sources are still shown so the reader can check them"


def test_a_sourced_quantity_is_kept(fake_backend, make_ctx):
    """The gate must not swallow numbers that ARE in the passages."""
    from sahel_sage.retrieval.evidence import Citation
    from sahel_sage.retrieval.rank import RRF_K

    two_legs = 2.0 / (RRF_K + 1)
    sourced = [
        Citation(
            doc_id="d1",
            title="Storage Manual",
            org="FAO",
            section="Drying",
            text="Dry the grain to 12% moisture before putting it into the store.",
            score=two_legs,
        )
    ]
    reply = GOOD.replace(
        "1. Uproot and burn the infected plants. [1]",
        "1. Dry the grain to 12% moisture first. [1]",
    ).replace("**Sources** [1][2]", "**Sources** [1]").replace(" [2]", " [1]")
    backend = fake_backend(reply)
    ctx = make_ctx(backend, sourced)

    r = answer("how dry should my grain be", "en", 4, ctx)

    assert not r.declined
    assert "12%" in r.raw_text


class TestResolveStatus:
    SUFFICIENT = EvidencePack(items=[], confidence=1.0, sufficient=True)
    INSUFFICIENT = EvidencePack(items=[], confidence=0.1, sufficient=False)

    def test_confident_pack_keeps_the_model_status(self):
        assert resolve_status(self.SUFFICIENT, Status.ANSWERED) is Status.ANSWERED

    def test_weak_pack_raises_the_floor(self):
        assert resolve_status(self.INSUFFICIENT, Status.ANSWERED) is Status.EVIDENCE_LIMITED

    def test_out_of_scope_is_never_downgraded(self):
        assert resolve_status(self.SUFFICIENT, Status.OUT_OF_SCOPE) is Status.OUT_OF_SCOPE
        assert resolve_status(self.INSUFFICIENT, Status.OUT_OF_SCOPE) is Status.OUT_OF_SCOPE

    def test_unparsed_status_is_treated_as_limited(self):
        assert resolve_status(self.SUFFICIENT, None) is Status.EVIDENCE_LIMITED


def test_numeric_gate_accepts_verified_reference_numbers():
    """The reference block is the one other legitimate home for a number.

    The gate once discarded a correct maize-storage answer for saying "13%
    moisture", a quantity that is test-enforced to appear in its cited FAO
    source, because the retrieved passages happened not to repeat it. Numbers
    from `data/reference/` are verified by construction and must not read as
    invented."""
    from sahel_sage.app.service import _reference_text
    from sahel_sage.training.numguard import unsupported_quantities

    answer = "Dry the grain to about 13% moisture before sealing the bags."
    irrelevant_passages = "Goats need clean water and dry bedding."
    assert unsupported_quantities(answer, irrelevant_passages),         "precondition: the number is truly absent from the passages"
    assert not unsupported_quantities(
        answer, irrelevant_passages + " " + _reference_text()
    )


def test_ui_file_has_no_stale_twin():
    """Two copies of the console UI exist: app/ui/ (the visible one people
    edit) and src/sahel_sage/app/ui/ (the one the server actually serves).
    A full round of UX fixes once went only to the visible copy, and the
    tester kept seeing the old page, worse than a bug, an invisible one.
    They must be byte-identical."""
    import sahel_sage.app.api as api
    from sahel_sage.core.config import repo_root

    visible = (repo_root() / "app" / "ui" / "index.html").read_bytes()
    served = api.UI_FILE.read_bytes()
    assert visible == served
