import json

from sahel_sage.core.prompts import ASSISTANT_CUE, USER_CUE, system_prompt
from sahel_sage.training.render import drop_system_prompt, render_dataset, render_record

CHUNK = {"chunk_id": "doc:0:abc", "doc_id": "doc", "title": "Goat Manual",
         "org": "ILRI", "cluster": "livestock", "text": "Give oral rehydration salts."}
CHUNKS = {"doc:0:abc": CHUNK}

ANSWER = ("**Likely issue**\nWatery diarrhoea in goats. [1]\n\n"
          "**What to do**\n1. Give oral rehydration salts. [1]\n\n"
          "**Caution**\nCall a vet if it lasts two days.\n\n**Sources** [1]")

GROUNDED = {"id": "doc:0:abc#0", "kind": "grounded_chunk",
            "q": "My goats have diarrhea?", "a": ANSWER,
            "meta": {"passage_ids": ["doc:0:abc"], "lang": "en"}}

BARE = {"id": "doc:0:abc#0:bare", "kind": "bare",
        "q": "How do I treat goat diarrhoea?", "a": ANSWER.split("\n\n**Sources**")[0],
        "meta": {"passage_ids": [], "lang": "en"}}

MULTI = {
    "id": "doc:0:abc#0:mt", "kind": "multi_turn",
    "turns": [
        {"q": "My goats have diarrhea?", "a": "One question first: are they still eating?"},
        {"q": "They eat less but still drink.", "a": "**Likely issue**\nDehydration.\n\nGive salts."},
    ],
    "meta": {"passage_ids": [], "lang": "en"},
}


def test_raw_grounded_matches_inference_format():
    r = render_record(GROUNDED, "raw", CHUNKS)
    assert r["prompt"].endswith("SAHEL SAGE:\n")
    assert "[1] Goat Manual — ILRI" in r["prompt"]
    assert "EXTRACTS FROM THE OFFLINE LIBRARY" in r["prompt"]
    assert r["completion"].startswith("**Likely issue**")
    assert "<think>" not in r["prompt"] + r["completion"]


def test_grounded_rows_never_lose_the_system_prompt():
    """The app path is the one we have measured as working; dropout must not
    touch it."""
    assert not drop_system_prompt(GROUNDED, rate=1.0)
    for rate in (0.0, 0.2, 1.0):
        r = render_record(GROUNDED, "raw", CHUNKS, dropout=rate)
        assert r["prompt"].startswith(system_prompt(True, "en"))


def test_chatml_bakes_empty_think_in_prompt_never_completion():
    r = render_record(GROUNDED, "chatml", CHUNKS)
    assert r["prompt"].rstrip().endswith("</think>")
    assert "<think>" not in r["completion"]
    assert r["completion"].endswith("<|im_end|>")


# --------------------------------------------------------------------------
# the judge path (ADR-005)
# --------------------------------------------------------------------------


def test_bare_rows_have_no_system_prompt_and_no_cues():
    r = render_record(BARE, "raw", {})
    assert r["prompt"] == "How do I treat goat diarrhoea?"
    assert USER_CUE not in r["prompt"]
    assert ASSISTANT_CUE not in r["prompt"]
    assert not r["prompt"].startswith(system_prompt(False, "en")[:40])
    assert "**Likely issue**" in r["completion"]


def test_multi_turn_renders_history_and_trains_only_the_last_turn():
    r = render_record(MULTI, "raw", {})
    prompt = r["prompt"]
    # The system prompt carries ONE embedded refusal demonstration that uses the
    # real cues (that realism is the point of an in-context demonstration), so
    # conversation turns are counted after the system prompt ends at the
    # language line.
    turns_part = prompt.split("Reply in English.", 1)[1]
    assert prompt.count(USER_CUE) == turns_part.count(USER_CUE) + 1
    assert turns_part.count(USER_CUE) == 2
    assert prompt.index("My goats have diarrhea?") < prompt.index("They eat less but still drink.")
    assert "One question first: are they still eating?" in prompt  # earlier turn is context
    assert prompt.endswith(ASSISTANT_CUE + "\n")
    assert r["completion"] == MULTI["turns"][-1]["a"]


def test_multi_turn_needs_at_least_two_turns():
    assert render_record(dict(MULTI, turns=[{"q": "hi", "a": "hello"}]), "raw", {}) is None


def test_system_prompt_dropout_keeps_the_cues():
    rec = dict(GROUNDED, id="x:cb", kind="closed_book",
               meta={"passage_ids": [], "lang": "en"})
    r = render_record(rec, "raw", {}, dropout=1.0)
    assert not r["prompt"].startswith(system_prompt(False, "en")[:40])
    assert r["prompt"].startswith(USER_CUE)
    assert r["prompt"].endswith(ASSISTANT_CUE + "\n")


def test_dropout_is_deterministic_and_roughly_the_requested_rate():
    rows = [{"id": f"row-{i}", "kind": "closed_book"} for i in range(2000)]
    first = [drop_system_prompt(r, 0.2) for r in rows]
    assert first == [drop_system_prompt(r, 0.2) for r in rows]  # stable across calls
    assert 0.15 < sum(first) / len(first) < 0.25
    assert sum(drop_system_prompt(r, 0.0) for r in rows) == 0


# --------------------------------------------------------------------------
# replay + plumbing
# --------------------------------------------------------------------------


def test_replay_arc_split():
    rec = {"kind": "replay_arc",
           "text": "Question: Why is the sky blue?\nAnswer: Rayleigh scattering."}
    r = render_record(rec, "raw", {})
    assert r["prompt"] == "Question: Why is the sky blue?\nAnswer:"
    assert "Rayleigh" in r["completion"]


def test_missing_passage_raises():
    bad = dict(GROUNDED, meta={"passage_ids": ["nope:0:xxx"], "lang": "en"})
    try:
        render_record(bad, "raw", CHUNKS)
        raise AssertionError("should have raised")
    except KeyError:
        pass


def test_render_dataset_roundtrip(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text("\n".join(json.dumps(r) for r in (GROUNDED, BARE, MULTI)) + "\n")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps(CHUNK) + "\n")
    out = tmp_path / "rendered.jsonl"
    stats = render_dataset(train, chunks, out, "raw")
    assert stats == {"style": "raw", "records_in": 3, "rendered": 3}
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    # `kind` rides along beside the two fields the trainer consumes. It exists so
    # a smoke run can sample WITHIN stratum: a head slice of a shuffled file can
    # cut the stratum under test to a handful of rows, and the run then proves
    # nothing while still costing GPU hours.
    assert all(set(row) == {"prompt", "completion", "kind"} for row in rows)
    assert all(row["kind"] for row in rows)


# --- per-row reference facts (Step 1) ---------------------------------------
#
# The full fifteen-fact block is 1,521 of the 1,892 constant prefix tokens on
# every closed-book row, which left the question at 1.6% of the input. The
# recall probe measured what a model does with that ratio: it reproduced the
# 65-character opening every refusal shares and diverged the moment the answer
# had to depend on the question. These tests pin the fix and, more importantly,
# the two ways it can silently go wrong.


def test_a_row_that_names_its_fact_carries_only_that_fact():
    """`reference_topic` rows carry their fact id, so no guessing is needed.

    A quarter of rows keep the whole block by design (see the test below), so
    this walks ids until it finds one on the subset path rather than pinning a
    single id whose hash could flip with any change to the bucketing.
    """
    from sahel_sage.training.render import facts_for

    for i in range(200):
        rec = {"id": f"ref:striga:{i}", "kind": "reference_topic",
               "q": "What is striga?",
               "meta": {"topic": "striga", "cluster": "reference", "lang": "en"}}
        facts = facts_for(rec)
        if facts is not None:
            assert len(facts) == 1 and facts[0].id == "striga"
            return
    raise AssertionError("every id kept the full block; the subset path is dead")


def test_a_row_with_no_usable_metadata_keeps_the_whole_block():
    """The default must be "all fifteen", never a guess.

    Stripping the one fact an answer depends on turns a grounded target into an
    unsupported one, a defect, where a slightly long prompt is merely a cost.
    Rows with no cluster and no topic are exactly where a selector would be
    guessing, so it must decline to.
    """
    from sahel_sage.training.render import facts_for

    assert facts_for({"id": "x", "kind": "closed_book"}) is None


def test_some_rows_keep_the_full_block_so_inference_stays_in_distribution():
    """At inference the template always emits all fifteen facts.

    Shortening every training row would trade one distribution mismatch for
    another, the exact mistake being fixed, where only 22% of rows matched the
    judged format. A deterministic slice therefore keeps the full block.
    """
    from sahel_sage.training.render import facts_for

    recs = [{"id": f"closed:{i}", "kind": "closed_book", "q": "How do I store maize?",
             "meta": {"cluster": "pest", "lang": "en"}} for i in range(400)]
    full = sum(facts_for(r) is None for r in recs)
    assert 60 < full < 160, f"{full}/400 rows kept the full block; expected ~25%"


def test_the_subset_choice_is_stable_across_rebuilds():
    from sahel_sage.training.render import facts_for

    rec = {"id": "closed:7", "kind": "closed_book", "q": "How do I store maize?",
           "meta": {"cluster": "pest", "lang": "en"}}
    a, b = facts_for(rec), facts_for(rec)
    assert (a is None) == (b is None)
    if a is not None:
        assert [f.id for f in a] == [f.id for f in b]


def test_dropping_the_system_prompt_leaves_no_debris_when_facts_are_subset():
    """The trap: `_raw_prompt` strips the system prompt by LENGTH.

    `render_raw` and the `len(system_prompt(...))` slice must receive the
    identical subset. Give one the subset and the other the full block and the
    slice lands mid-sentence, leaving prompt debris that still trains, on
    garbage, and that no other test looks for. This row is chosen to be one
    the dropout actually fires on.
    """
    from sahel_sage.training.render import drop_system_prompt, render_record

    rec = dict(BARE, kind="closed_book", id="doc:0:abc#0:cb",
               meta={"passage_ids": [], "lang": "en", "cluster": "livestock"})
    # find an id this deterministic dropout actually drops
    for i in range(200):
        candidate = dict(rec, id=f"cb:{i}")
        if drop_system_prompt(candidate):
            rec = candidate
            break
    else:
        raise AssertionError("dropout never fired; cannot exercise the strip")

    r = render_record(rec, "raw", CHUNKS)
    # A correct strip starts exactly at the question cue. Debris shows up as
    # any leftover of the block or the instructions before it.
    assert r["prompt"].startswith(USER_CUE), f"debris before the cue: {r['prompt'][:80]!r}"
    assert "WHAT YOU KNOW ABOUT SAHELIAN FARMING" not in r["prompt"]
    assert "You are Sahel Sage" not in r["prompt"]


def test_the_shipped_template_still_carries_every_fact():
    """Whatever training does, inference must see all fifteen."""
    from sahel_sage.core.prompts import chat_template
    from sahel_sage.core.reference import load_reference

    t = chat_template("en")
    for fact in load_reference():
        assert fact.topic in t, f"{fact.id} missing from the shipped template"
