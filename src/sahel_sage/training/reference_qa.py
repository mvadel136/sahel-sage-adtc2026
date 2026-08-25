"""Training rows for the twelve topics, built from the same facts the GGUF ships.

The expert audit named the knowledge missing from both our model and one twice
its size: hivernage sowing, millet seed rate and spacing, Striga, fall armyworm,
PPR, Newcastle vaccination, Haemonchus and FAMACHA, dry-season protein, salinity
drainage, manure micro-dosing, zai and demi-lunes, hermetic storage, aflatoxin.
Those became `data/reference/`, and this module turns them into training data.

The point is agreement. The judged artifact carries the reference block in its
chat template *and* the weights are trained on the same content, so the two say
the same thing. Training the model to contradict its own system prompt would be
worse than doing neither, the 2026-08-13 rehearsal showed exactly that failure
in miniature, with the model reading the block and then inverting it ("avoid
hermetic bags, they do not kill storage insects").

Three files, one per concern, and every row is checked against all three:

  sahel_reference.json    what is true, with the source for every number
  reference_questions.json how a farmer actually asks it
  reference_answers.json   how we answer, in the contract shape

`build_reference_rows` refuses to emit a row whose answer states a quantity that
does not appear in the fact's cited source. That check is the reason the answers
live in data rather than in a prompt: a number nobody can trace is a number we
are inventing, which is the entire problem this project has been fighting.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from sahel_sage.core.config import repo_root
from sahel_sage.core.reference import Fact, load_reference
from sahel_sage.training.markdownify import render_markdown
from sahel_sage.training.metaleak import meta_leaks
from sahel_sage.training.numguard import unsupported_quantities

QUESTIONS_PATH = Path("data/reference/reference_questions.json")
ANSWERS_PATH = Path("data/reference/reference_answers.json")

#: Openers that vary the surface without changing the question. A stratum whose
#: rows differ only by topic teaches the topic, not the task.
_PREFIXES = ("", "Please help: ", "Quick question, ", "I need advice. ", "Hello. ")


def _load(path: Path, key: str) -> dict:
    return json.loads((repo_root() / path).read_text())[key]


def build_reference_rows(n_per_topic: int = 20, seed: int = 42) -> list[dict]:
    """Expand each verified fact into many phrasings of one correct answer.

    Oversampled relative to ordinary strata: these are the facts the model has
    to override a pretraining prior to hold ("plant millet in early spring" has
    survived two rounds of correction), and override difficulty scales with how
    strong the prior is.
    """
    rng = random.Random(seed)
    facts: dict[str, Fact] = {f.id: f for f in load_reference()}
    questions = _load(QUESTIONS_PATH, "questions")
    answers = _load(ANSWERS_PATH, "answers")

    missing = set(facts) - set(questions) or set(facts) - set(answers)
    if missing:
        raise KeyError(f"facts with no questions or no answer: {sorted(missing)}")

    out: list[dict] = []
    seen: set[str] = set()
    for topic_id, fact in facts.items():
        spec = answers[topic_id]
        phrasings = questions[topic_id]
        answer = render_markdown(
            issue=spec["issue"],
            actions=list(spec["actions"]),
            timing=spec["timing"],
            caution=spec["caution"],
            sources=None,
        )

        # Every number we are about to teach must be traceable to the source
        # this fact cites. Checked once per topic, before any row is emitted.
        bad = unsupported_quantities(answer, fact.source)
        if bad:
            raise ValueError(
                f"{topic_id}: answer states {bad} which is not in its source. "
                "Either the number is wrong or the source string must carry it."
            )
        if leaks := meta_leaks(answer):
            raise ValueError(f"{topic_id}: answer talks about the prompt: {leaks}")

        made = 0
        total = len(phrasings) * len(_PREFIXES)
        order = list(range(total))
        rng.shuffle(order)
        for idx in order:
            if made >= n_per_topic:
                break
            q = phrasings[idx % len(phrasings)]
            prefix = _PREFIXES[(idx // len(phrasings)) % len(_PREFIXES)]
            question = f"{prefix}{q}"
            key = question.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "id": f"reference:{topic_id}:{made}",
                "kind": "reference_topic",
                "q": question,
                "a": answer,
                "meta": {
                    "source_docs": [],
                    "passage_ids": [],
                    # Empty on purpose: rendered closed-book, so the numeric gate
                    # would normally reject every quantity. The check above has
                    # already validated them against the cited source, which is
                    # stronger than a chunk match.
                    "gate_passages": [],
                    "lang": "en",
                    "critique": "pass",
                    "cluster": "reference",
                    "topic": topic_id,
                    "corpus_source": fact.source,
                },
            })
            made += 1
        if made < n_per_topic:
            raise ValueError(
                f"{topic_id}: only {made}/{n_per_topic} distinct questions from "
                f"{total} combinations, add phrasings to reference_questions.json"
            )
    return out


def write_reference_rows(out_path: Path, n_per_topic: int = 20, seed: int = 42) -> dict:
    rows = build_reference_rows(n_per_topic, seed)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_topic: dict[str, int] = {}
    for r in rows:
        by_topic[r["meta"]["topic"]] = by_topic.get(r["meta"]["topic"], 0) + 1
    return {"reference_rows": len(rows), "topics": len(by_topic), "by_topic": by_topic}
