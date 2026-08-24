"""The GGUF chat template MUST reproduce render_raw exactly.

This is the highest-stakes invariant in the project: judges chat with the bare
model through their own client, which applies the template embedded in the
GGUF. If the template drifts from the training rendering, every judged answer
is generated off-distribution and no benchmark we run would reveal it
(ADR-005).
"""

import pytest

from sahel_sage.core.prompts import EvidenceItem, chat_template, render_raw

jinja2 = pytest.importorskip("jinja2")


@pytest.fixture
def tpl():
    return jinja2.Template(chat_template())


def test_bare_question_matches_training_format(tpl):
    q = "How do I treat diarrhea in goats?"
    assert tpl.render(messages=[{"role": "user", "content": q}],
                      add_generation_prompt=True) == render_raw(q, [])


def test_multi_turn_matches_training_format(tpl):
    q1, a1, q2 = "Why is my millet yellow?", "**Likely issue**\nNitrogen shortage.", "And my sorghum?"
    rendered = tpl.render(
        messages=[{"role": "user", "content": q1},
                  {"role": "assistant", "content": a1},
                  {"role": "user", "content": q2}],
        add_generation_prompt=True,
    )
    assert rendered == render_raw(q2, [], history=[(q1, a1)])


def test_caller_system_prompt_appends_and_ours_survives(tpl):
    """A judge's chat UI may quietly send its own system message ("You are a
    helpful assistant."). Under the old replace rule that single line deleted
    the reference block and every safety instruction from the judged
    conversation. The caller's text is honoured — appended after ours — but
    the facts and the safety rules always render."""
    out = tpl.render(messages=[{"role": "system", "content": "CUSTOM"},
                               {"role": "user", "content": "hi"}],
                     add_generation_prompt=True)
    assert "CUSTOM" in out
    assert "WHAT YOU KNOW ABOUT SAHELIAN FARMING" in out
    assert "You are Sahel Sage" in out
    assert out.index("You are Sahel Sage") < out.index("CUSTOM")


def test_extracts_switch_to_sourced_system_prompt(tpl):
    item = EvidenceItem(n=1, title="T", org="O", section="S", text="body text")
    user = render_raw("q", [item]).split("\n\n", 1)[1].rsplit("\n\nSAHEL SAGE:", 1)[0]
    out = tpl.render(messages=[{"role": "user", "content": user}], add_generation_prompt=True)
    assert "mark each sentence that uses one with its number" in out


def test_empty_messages_do_not_crash(tpl):
    assert isinstance(tpl.render(messages=[], add_generation_prompt=True), str)


def test_template_uses_no_exotic_filters():
    """minja implements only the filters mainstream templates use."""
    assert "|" not in chat_template().replace("{%-", "").replace("-%}", "")
