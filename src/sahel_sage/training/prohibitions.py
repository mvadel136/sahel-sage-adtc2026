"""Training rows that teach the model to refuse what it must never answer.

The app refuses these questions before the model is ever called
(`inference.safety.screen`), so why train them at all? Because **the judged
artifact is the bare GGUF**. Competition judges chat with the model file
directly; no deny-list, no retrieval, no application code stands in front of it.
The layer in `inference.safety` protects the farmer using the app. This module
is what protects the farmer talking to the model.

Two design rules, both learned the hard way:

**One source of truth for the refusal text.** The `a` field is
`Prohibition.response` verbatim, so a change to what the app says is
automatically a change to what the model is trained to say. When they were
allowed to drift in earlier rounds, the model learned a paraphrase that dropped
the part naming who to ask.

**Qualifiers must not move the answer.** Every generated question is a
prohibited question wearing a different excuse — "if I wash it first", "just
this once", "he's careful", "the vet is far away". The expert audit found the
round-4 model taking exactly these escape hatches: offered "if I wash it", it
answered *use the pesticide container for drinking water*. Training on the
excuse-free phrasing alone teaches the model to refuse a question no farmer
actually asks.

Every generated question is asserted to trip `screen()`, so this module cannot
silently drift out of the prohibited class it claims to cover.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from sahel_sage.inference.safety import PROHIBITIONS, screen

#: Excuses that must not work. Each is a real thing a farmer says, and each is
#: an invitation for a model to negotiate. Empty string = the plain phrasing.
QUALIFIERS: tuple[str, ...] = (
    "",
    "I know it is not ideal, but ",
    "Just this once, ",
    "It is an emergency — ",
    "The extension agent is not answering. ",
    "My neighbour says it is fine. ",
    "I have done it before with no problem. ",
    "I will be very careful. ",
    "I cannot afford the proper way. ",
    "The shop is closed and I need to act today. ",
)

#: Ways of asking, before the specific subject. Mixed with QUALIFIERS by a
#: mixed-radix walk so no two rows repeat a (qualifier, opener, question) triple
#: — a stratum that collapses to a handful of distinct strings teaches nothing,
#: which happened three times to `abstain_scope` in earlier rounds.
OPENERS: tuple[str, ...] = (
    "",
    "please tell me, ",
    "quick question: ",
    "I need to know — ",
    "can you help: ",
)

#: Per-prohibition question phrasings. These must each trip `screen()`; the
#: builder asserts it rather than trusting it.
PHRASINGS: dict[str, tuple[str, ...]] = {
    "pesticide_rate": (
        "how many millilitres of pesticide should I mix per litre of water?",
        "what is the right dose of insecticide for my sprayer?",
        "how much chemical do I put in a 15 litre knapsack?",
        "what is the dilution rate for this herbicide?",
        "how much deltamethrin per litre for my cowpeas?",
        "how strong should I make the spray for my tomatoes?",
        'the shopkeeper wrote something on the bag but I cannot read it, how much do I use?',
        'my neighbour uses two caps per sprayer, should I do the same?',
        'I have a small bottle and a big tank, how do I work out the amount?',
        'is one spoon per litre about right for aphids?',
        'how do I make the spray stronger if the pests do not die?',
        'what quantity of product for one hectare of cowpea?',
    ),
    "preharvest_interval": (
        "I sprayed my tomatoes yesterday. can I harvest and sell them today?",
        "how long after spraying before I can pick the vegetables?",
        "I treated the crop three days ago, is that enough time before I sell?",
        "when is it safe to eat produce after spraying it?",
        "the market is tomorrow and I sprayed this morning. can I still sell?",
        'the buyer is coming this afternoon and I sprayed on Monday, is that alright?',
        'my tomatoes look ready but I treated them recently, what do I do?',
        'how many days should pass between spraying and eating the vegetables?',
        'can my family eat from the field I sprayed this week?',
        'I need money urgently, can I sell the crop now even though I sprayed?',
    ),
    "veterinary_drug": (
        "should I give my goat antibiotics for diarrhea?",
        "what dose of oxytetracycline for a sheep?",
        "how much dewormer should I give a young calf?",
        "which antibiotic works best for sick chickens?",
        "how many millilitres do I inject into a goat?",
        'the animal pharmacy has several bottles, which one do I buy for a coughing goat?',
        'my sheep is scouring, what injection should I give and how much?',
        'how many days of antibiotic does a goat need?',
        'someone told me to give penicillin, is that the right one?',
        'what treatment do I give a calf with a swollen joint?',
    ),
    "human_medicine_for_animals": (
        "can I use human paracetamol on a sick cow?",
        "is it safe to give my goat ibuprofen for pain?",
        "can I give the calf tablets from the pharmacy?",
        "my cow is in pain, can I use my own medicine?",
        "would aspirin help a sick sheep?",
        'I have tablets left from the clinic, can the goat have them?',
        'my cow will not stand and I have some medicine at home, should I try it?',
        'is there any human medicine that is safe for animals?',
        'the pharmacy in the village only sells for people, can I use that?',
        'half a tablet for a small goat, would that hurt it?',
    ),
    "pesticide_container_reuse": (
        "can I use an empty pesticide container to carry drinking water?",
        "is it fine to store milk in a washed pesticide bottle?",
        "I rinsed the pesticide drum well. can I keep grain in it?",
        "can the old chemical jerrycan be used for cooking oil?",
        "what can I reuse an empty pesticide container for?",
        'the empty drums are useful, what can I safely do with them?',
        'my children carry water in an old chemical can, is that a problem?',
        'I boiled the container, does that make it safe?',
        'can I keep seed in the empty spray drum?',
        'what should I do with the containers after I finish spraying?',
    ),
    "minor_applying_pesticide": (
        "my son is 12. can he spray the field while I am away?",
        "can my daughter help me mix the chemical?",
        "he is only 14 but he has done it before. can he spray?",
        "is it alright for the children to carry the sprayer?",
        "can a boy of 15 do the spraying for me?",
        'the school holiday is on and my sons want to help in the field, what can they do?',
        'my nephew is fifteen and strong, can he carry the sprayer?',
        'who in the family is allowed to handle the chemicals?',
        'at what age can someone start spraying?',
        'the boys mix it and I spray it, is that alright?',
    ),
    "banned_pesticide": (
        "is it okay to use paraquat on my weeds?",
        "someone is selling endosulfan cheaply. should I buy it?",
        "can I use monocrotophos on my cowpeas?",
        "what do you think of using DDT for storage pests?",
        "is aluminium phosphide good for my grain store?",
        'the market sells a cheap powder that kills everything, should I buy it?',
        'an old farmer told me to use DDT for the grain store, is that still done?',
        'what is the strongest thing I can buy for my weeds?',
        'someone offered me paraquat cheap, is it worth it?',
        'which chemical kills the pests fastest?',
    ),
}


def _sentence_case(text: str) -> str:
    """Capitalise the first letter of each sentence.

    A qualifier ends in a full stop and the opener follows it, so the naive
    "upper-case index 0" left rows reading "I have done it before. please tell
    me, ...". Farmers type untidily and some of that noise is welcome, but not
    noise that only exists because of how we glued three pools together.
    """
    out = []
    capitalise = True
    for ch in text:
        out.append(ch.upper() if capitalise and ch.isalpha() else ch)
        if ch in ".?!":
            capitalise = True
        elif not ch.isspace():
            capitalise = False
    return "".join(out)


def build_prohibition_rows(n_per_rule: int = 60, seed: int = 42) -> list[dict]:
    """Expand each prohibition into many excuse-carrying phrasings.

    Oversampled relative to ordinary strata on purpose: a refusal has to hold
    against a question actively arguing for the opposite, and the base model's
    prior is to be helpful.
    """
    rng = random.Random(seed)
    rules = {p.id: p for p in PROHIBITIONS}
    out: list[dict] = []
    seen: set[str] = set()

    for rule_id, phrasings in PHRASINGS.items():
        rule = rules.get(rule_id)
        if rule is None:
            raise KeyError(f"PHRASINGS has {rule_id!r} but inference.safety does not")
        made = 0
        # mixed-radix walk over (phrasing, qualifier, opener) so the three pools
        # multiply instead of cycling in lockstep
        total = len(phrasings) * len(QUALIFIERS) * len(OPENERS)
        order = list(range(total))
        rng.shuffle(order)
        for idx in order:
            if made >= n_per_rule:
                break
            p = phrasings[idx % len(phrasings)]
            q = QUALIFIERS[(idx // len(phrasings)) % len(QUALIFIERS)]
            o = OPENERS[(idx // (len(phrasings) * len(QUALIFIERS))) % len(OPENERS)]
            question = _sentence_case(f"{q}{o}{p}".strip())
            key = question.lower()
            if key in seen:
                continue
            # the whole point is that the excuse does not change the class
            assert screen(question) is not None, (
                f"generated question is not caught by screen(): {question!r}"
            )
            seen.add(key)
            out.append({
                "id": f"prohibition:{rule_id}:{made}",
                "kind": "prohibition",
                "q": question,
                "a": rule.response,
                "meta": {
                    "source_docs": [],
                    "passage_ids": [],
                    "gate_passages": [],
                    "lang": "en",
                    "critique": "pass",
                    "cluster": "safety",
                    "rule": rule_id,
                    "rationale": rule.rationale,
                },
            })
            made += 1
        if made < n_per_rule:
            raise ValueError(
                f"{rule_id}: only {made}/{n_per_rule} distinct questions available "
                f"from {total} combinations — widen its PHRASINGS pool"
            )
    return out


def write_prohibition_rows(out_path: Path, n_per_rule: int = 60, seed: int = 42) -> dict:
    rows = build_prohibition_rows(n_per_rule, seed)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_rule: dict[str, int] = {}
    for r in rows:
        by_rule[r["meta"]["rule"]] = by_rule.get(r["meta"]["rule"], 0) + 1
    return {"prohibition_rows": len(rows), "rules": len(PHRASINGS), "by_rule": by_rule}
