"""Round-3 ALL-CAPS answers -> the round-4 markdown contract (ADR-005)."""

import json

from sahel_sage.training.markdownify import markdownify, markdownify_pairs

OLD = """LIKELY ISSUE: Downy mildew on millet seedlings. [1]
ACTIONS:
1. Remove and burn infected plants. [1]
2. Rotate with cowpea next season. [2]
TIMING: Act within a week of first symptoms. [1]
CAUTION: If more than half the field is affected, call the extension agent.
SOURCES: [1][2]
STATUS: ANSWERED"""


def test_headings_replace_labels():
    md = markdownify(OLD)
    assert md.startswith("**Likely issue**\nDowny mildew on millet seedlings. [1]")
    assert "**What to do**\n1. Remove and burn infected plants. [1]" in md
    assert "**Timing**\nAct within a week" in md
    assert "**Caution**\nIf more than half" in md
    assert md.rstrip().endswith("**Sources** [1][2]")


def test_status_line_is_dropped_entirely():
    md = markdownify(OLD)
    assert "STATUS" not in md
    assert "ANSWERED" not in md


def test_no_all_caps_labels_survive():
    md = markdownify(OLD)
    for label in ("LIKELY ISSUE:", "ACTIONS:", "TIMING:", "CAUTION:", "SOURCES:"):
        assert label not in md


def test_citation_markers_are_preserved():
    md = markdownify(OLD)
    assert md.count("[1]") == OLD.count("[1]") == 4
    assert md.count("[2]") == OLD.count("[2]") == 2


def test_sources_none_produces_no_sources_section():
    md = markdownify(OLD.replace("SOURCES: [1][2]", "SOURCES: NONE"))
    assert "**Sources**" not in md
    assert md.rstrip().endswith("call the extension agent.")


def test_unparseable_input_returns_none():
    assert markdownify("Just water the plants more, they will be fine.") is None
    assert markdownify("LIKELY ISSUE: something is wrong.") is None  # no actions
    assert markdownify("") is None


def test_inline_run_on_contract_is_split_not_swallowed():
    """The teacher sometimes emitted the whole contract on one line; left
    inline the tail survived into the markdown target as ALL-CAPS residue."""
    run_on = (
        "LIKELY ISSUE: Corral cattle to manure the soil.\n"
        "ACTIONS:\n1. Rotate the corrals between fields. TIMING: After harvest. "
        "CAUTION: Keep corrals away from crops. SOURCES: [1]. STATUS: ANSWERED"
    )
    md = markdownify(run_on)
    assert "STATUS" not in md and "TIMING:" not in md
    assert "**Timing**\nAfter harvest." in md
    assert "**Caution**\nKeep corrals away from crops." in md


def test_actions_capped_at_five():
    actions = "\n".join(f"{i}. step {i}" for i in range(1, 9))
    md = markdownify(OLD.replace(
        "1. Remove and burn infected plants. [1]\n2. Rotate with cowpea next season. [2]", actions
    ))
    assert "5. step 5" in md and "6. step 6" not in md


def test_markdownify_pairs_roundtrip(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join([
        json.dumps({"id": "a", "q": "q?", "a": OLD, "meta": {"lang": "en"}}),
        json.dumps({"id": "b", "q": "q2?", "a": "no contract here at all"}),
    ]) + "\n")
    out = tmp_path / "out.jsonl"
    stats = markdownify_pairs(src, out)

    assert stats == {"in": 2, "converted": 1, "dropped": 1}
    rec = json.loads(out.read_text())
    assert rec["id"] == "a"
    assert rec["a"].startswith("**Likely issue**")
    assert rec["meta"]["format"] == "markdown_v4"
