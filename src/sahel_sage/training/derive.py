"""Derive the non-grounded training strata from existing data — no teacher calls.

Round-4 rewrite (ADR-005). The judges chat with the **bare model** through
their own client: no retrieval app, no evidence block, and — if the chat
template is ever bypassed — no system prompt and no cues either. Round 3
trained 9,082 rows that all carried system prompt + evidence + cues, so a bare
question was off-distribution and the model rambled past its answer until the
token cap. These generators build the missing behaviours out of material we
already trust:

- ``closed_book``   — a grounded pair with the evidence removed. Answers
  **confidently** from its own knowledge, no citations, no hedging, and never
  mentions a "library" the judge cannot see. (v3 wrongly made these
  EVIDENCE_LIMITED, which reads as evasion on every judge question.)
- ``abstain_limited`` — a real question shown MISMATCHED extracts: one sentence
  saying the extracts do not cover it, then general practice, never a citation.
- ``abstain_scope`` — out-of-domain questions, short friendly redirect.
- ``greeting``     — the judge's literal first turn ("hi", "what can you do?").
  No contract headings: a greeting answered with "**Likely issue**" is a
  first-impression failure.
- ``bare``         — existing pairs rendered with NO system prompt and NO cue.
- ``multi_turn``   — 2-3 turn conversations (clarification / refinement /
  drill-down / topic switch) where only the last assistant turn is trained.

Two dataset-wide invariants are enforced here rather than hoped for:

*Anti-repetition.* Every boilerplate sentence is sampled from a list of
variants (`CAUTION_VARIANTS` and friends). One repeated closing sentence across
thousands of rows is exactly the repetition hook that made the round-3 model
loop under greedy decoding.

*Dedup safety.* The mixer dedups on normalized question and exact answer within
a kind, and its stats count rows *before* dedup — so a stratum can silently
collapse (v2 shipped 1 scope row of 240; v3 shipped 120 of 450). Every
generator here therefore produces distinct questions AND distinct answers by
construction, and the dataset build asserts the realized counts.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from sahel_sage.core.prompts import SECTION_CAUTION, SECTION_ISSUE, Status
from sahel_sage.inference.contract import infer_status, parse
from sahel_sage.training.markdownify import render_markdown
from sahel_sage.training.normalize import CAUTION_VARIANTS, sample_caution

_CITE = re.compile(r"\s*\[\d{1,2}\]")
_SENT = re.compile(r"(?<=[.!?])\s+")
_SOURCES_BLOCK = re.compile(r"\n{0,2}\*\*Sources\*\*.*$", re.DOTALL | re.IGNORECASE)

#: Phrases that would break the judge-path illusion: the judge sees no library.
_LEAKS = ("offline library", "the extracts", "these extracts", "the passages", "the documents")


# --------------------------------------------------------------------------
# small text helpers
# --------------------------------------------------------------------------


def strip_cites(text: str) -> str:
    return _CITE.sub("", text).strip()


def uncite(answer: str) -> str:
    """Drop citation markers AND the whole `**Sources**` block.

    History turns inside a multi-turn conversation carry no evidence, so a
    `**Sources**` heading with nothing after it (what `strip_cites` alone
    leaves behind) would teach the model to emit an empty section.
    """
    return _SOURCES_BLOCK.sub("", strip_cites(answer)).rstrip()


def trim_words(text: str, max_words: int) -> str:
    """Cut to whole sentences within a word budget (never mid-sentence)."""
    out: list[str] = []
    used = 0
    for sent in _SENT.split(text.strip()):
        n = len(sent.split())
        if out and used + n > max_words:
            break
        out.append(sent)
        used += n
    return " ".join(out) if out else text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def _pick(options, i: int):
    return options[i % len(options)]


def cluster(rec: dict) -> str:
    return rec.get("meta", {}).get("cluster", "")


#: A clarifying question only makes sense over a *problem* ("my goats have
#: diarrhoea"), never over a lookup ("what distinguishes this sheep breed?").
#: Asking "are they still eating?" about a breed description teaches the model
#: that its own question need not relate to what was asked.
_PROBLEM_WORDS = (
    "what is wrong", "why are", "why is", "why do my", "dying", "died", "sick",
    "disease", "attacked", "damage", "wilting", "yellowing", "spots on", "rotting",
    "not growing", "poor yield", "diarrh", "coughing", "swollen", "infested",
    "eating my", "losing weight", "lame", "fever", "symptom",
)


def looks_like_a_problem(question: str) -> bool:
    """True for symptom questions, where "are they still eating?" makes sense.

    Deliberately narrow: a loose match ("my stored maize") produced
    conversations where the model asked about leaf spots on a compost question,
    which teaches it that its own clarifying question need not relate to
    anything.
    """
    low = question.lower()
    return any(w in low for w in _PROBLEM_WORDS)


# --------------------------------------------------------------------------
# out-of-scope
# --------------------------------------------------------------------------

SCOPE_ITEMS: list[tuple[str, str]] = [
    ("Who won the football world cup?", "Football results"),
    ("Which team will win the league this season?", "League predictions"),
    ("Can you write me a poem about the ocean?", "Poetry writing"),
    ("Write a short story for my daughter.", "Story writing"),
    ("What is the best phone to buy this year?", "Choosing a phone"),
    ("Which laptop should I buy for university?", "Choosing a computer"),
    ("How do I fix the engine of my motorbike?", "Motorbike repair"),
    ("My car will not start, what is wrong?", "Car repair"),
    ("Tell me about the history of ancient Rome.", "Ancient history"),
    ("Who built the pyramids of Egypt?", "The pyramids of Egypt"),
    ("What will the exchange rate be next month?", "Currency exchange rates"),
    ("Should I invest my savings in bitcoin?", "Personal investment"),
    ("How can I get a visa to travel to Europe?", "Visa paperwork"),
    ("What documents do I need for a passport?", "Travel documents"),
    ("What are the lyrics of the national anthem?", "Song lyrics"),
    ("Sing me a song in Arabic.", "Singing a song"),
    ("Who is the president of the United States?", "Foreign politics"),
    ("When are the next elections?", "Politics"),
    ("How do I create a website for my cousin's shop?", "Website building"),
    ("Write me a Python script to sort numbers.", "Computer programming"),
    ("Can you help me with my mathematics homework?", "Mathematics homework"),
    ("Explain quantum physics to me.", "Physics"),
    ("What medicine should I take for my headache?", "Human medicine"),
    ("My child has a fever, what should I give her?", "Medicine for a sick child"),
    ("How do I open a bank account?", "Banking"),
    ("What is the interest rate on a house loan?", "Bank loans"),
    ("Translate this letter into English for me.", "Translation"),
    ("What is the weather forecast for next week?", "Weather forecasting"),
    ("Tell me a joke.", "Jokes"),
    ("What time does the bus to Nouakchott leave?", "Transport timetables"),
]

#: 15 openings x 30 topics = 450 distinct normalized questions.
_SCOPE_PREFIXES = (
    "",
    "Please tell me: ",
    "I want to know - ",
    "Quick question: ",
    "Hello, ",
    "My friend asked me: ",
    "Can you tell me, ",
    "I am curious: ",
    "Someone told me to ask you: ",
    "Before I forget: ",
    "One more thing: ",
    "Sorry to bother you, but ",
    "While I have you: ",
    "Just wondering - ",
    "Off topic, but ",
)

#: Each contains an explicit "not a/about farming…" phrase so the inferred
#: status is OUT_OF_SCOPE (contract.infer_status).
_SCOPE_REFUSALS = (
    "{topic} is not a farming, livestock or rural livelihood question, so it is outside what I cover.",
    "I cannot help with {lower} — it is not a farming, livestock or land-management topic.",
    "{topic} sits outside my area: I am not a general assistant, only a farming and livestock advisor.",
    "That is a question about {lower}, not about farming, livestock, soil or water, so I have to pass.",
    "{topic} is not a farming question, and answering it well is somebody else's job, not mine.",
)

#: 15 variants -> 30 topics x 15 = 450 distinct answers.
_SCOPE_CAUTIONS = (
    "Ask me about planting, pests, livestock health, soil or storage and I will give you a practical answer.",
    "What I am good at is crops, animal health, soil fertility, water and post-harvest storage.",
    "Bring me a question about your field, your herd or your grain store and I can really help.",
    "I can advise on seed choice, sowing dates, fertiliser, pests, diseases and irrigation.",
    "Animal health, feeding, watering and housing for cattle, goats, sheep and poultry are my ground.",
    "If something is wrong with your crop or your animals, describe the symptoms and I will work through it.",
    "Soil fertility, erosion, composting and water harvesting are all things I can walk you through.",
    "I also cover harvesting, drying, storage losses and how to keep grain free of insects.",
    "Ask about vegetable gardening, tree planting or fodder production and I will give you steps you can act on.",
    "Questions about market gardening, poultry keeping or small ruminants are well within my area.",
    "Tell me your crop, your region and what you are seeing, and I will suggest what to do next.",
    "Pest and disease identification, and what to do about them without harming yourself, is my speciality.",
    "I can help you plan a season: what to sow, when to sow it, and how to prepare the land.",
    "Feed shortages, dry-season grazing and watering points are topics I can advise on.",
    "Anything about farming, herding, soil, water or rural livelihoods is fair game — ask away.",
)


def scope_answer(topic: str, refusal: str, caution: str) -> str:
    """Short friendly redirect. MUST vary: identical targets collapse under
    answer-dedup (v2 shipped 1 of 240; v3 shipped 120 of 450)."""
    issue = refusal.format(topic=topic, lower=topic[0].lower() + topic[1:])
    return f"{SECTION_ISSUE}\n{issue}\n\n{SECTION_CAUTION}\n{caution}"


def derive_abstain_scope(n: int, rng: random.Random | None = None) -> list[dict]:
    out = []
    for i in range(n):
        q, topic = SCOPE_ITEMS[i % len(SCOPE_ITEMS)]
        cycle = i // len(SCOPE_ITEMS)
        prefix = _pick(_SCOPE_PREFIXES, cycle)
        question = prefix + (q[0].lower() + q[1:] if prefix.endswith(("but ", ", ")) else q)
        out.append({
            "id": f"scope:{i}",
            "kind": "abstain_scope",
            "q": question,
            "a": scope_answer(topic, _pick(_SCOPE_REFUSALS, cycle), _pick(_SCOPE_CAUTIONS, cycle)),
            "meta": {"source_docs": [], "passage_ids": [], "lang": "en",
                     "critique": "pass", "derived": "abstain_scope"},
        })
    return out


# --------------------------------------------------------------------------
# closed book / bare — confident, uncited answers
# --------------------------------------------------------------------------


def _uncited_answer(rec: dict, caution: str) -> str | None:
    """Grounded markdown answer -> the same advice with no evidence attached."""
    c = parse(rec["a"]).contract
    issue = strip_cites(c.likely_issue)
    actions = [strip_cites(a) for a in c.actions]
    if not issue or not actions:
        return None
    answer = render_markdown(
        issue=issue,
        actions=actions,
        timing=strip_cites(c.timing),
        caution=strip_cites(c.caution) or caution,
        sources=None,
    )
    low = answer.lower()
    if "[" in answer or any(leak in low for leak in _LEAKS):
        return None
    # The judge path must read as knowledge, not as an apology.
    if infer_status(answer) is not Status.ANSWERED:
        return None
    return answer


def _confident_rows(pairs: list[dict], kind: str, suffix: str, rng: random.Random) -> list[dict]:
    out = []
    for rec in pairs:
        caution = sample_caution(rng, rec["a"], rec.get("meta", {}).get("cluster", ""))
        answer = _uncited_answer(rec, caution)
        if answer is None:
            continue
        out.append({
            "id": rec["id"] + suffix,
            "kind": kind,
            "q": rec["q"],
            "a": answer,
            "meta": {**rec.get("meta", {}), "passage_ids": [],
                     "gate_passages": rec.get("meta", {}).get("passage_ids", []),
                     "derived": kind},
        })
    return out


def derive_closed_book(pairs: list[dict], n: int, rng: random.Random) -> list[dict]:
    """No evidence -> answer confidently from own knowledge (ADR-005).

    v3 made this stratum say "the offline library does not cover this question"
    whenever no evidence was present. In the app that is right; in the judges'
    sandbox — where there is no library — it reads as evasion on every single
    question, which is why it is reversed here.
    """
    return _confident_rows(rng.sample(pairs, min(n, len(pairs))), "closed_book", ":cb", rng)


def derive_bare_questions(pairs: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Same content as closed_book, but rendered with NO system prompt and NO
    cue (see training.render) — the shape the judge's client may actually send
    if the GGUF chat template is ignored."""
    return _confident_rows(rng.sample(pairs, min(n, len(pairs))), "bare", ":bare", rng)


# --------------------------------------------------------------------------
# abstain_limited — mismatched extracts
# --------------------------------------------------------------------------

_LIMITED_OPENERS = (
    "The extracts I was given do not cover this question.",
    "None of the extracts here deal with this problem.",
    "The passages provided are about something else, so they do not answer this.",
    "What I have been shown does not address this question.",
    "These extracts do not contain anything on this, so I will not cite them.",
    "The material at hand does not cover your case.",
    "Nothing in the extracts speaks to this situation.",
    "The extracts miss this topic entirely.",
)

_LIMITED_BRIDGES = (
    "From general practice:",
    "From general agricultural practice:",
    "Speaking from common practice instead:",
    "Working from standard practice:",
    "Here is what general practice says:",
    "From what is normally recommended:",
)


def derive_abstain_limited(pairs: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Question + wrong-cluster passages -> say so in one sentence, then answer
    from general practice, and never cite the extracts."""
    by_cluster: dict[str, list[dict]] = {}
    for rec in pairs:
        by_cluster.setdefault(rec.get("meta", {}).get("cluster", "?"), []).append(rec)
    clusters = [c for c in by_cluster if len(by_cluster[c]) > 10]
    if len(clusters) < 2:
        return []

    pool = rng.sample(pairs, min(n * 2, len(pairs)))
    out: list[dict] = []
    for i, q_rec in enumerate(pool):
        if len(out) >= n:
            break
        c_q = q_rec.get("meta", {}).get("cluster", "?")
        others = [c for c in clusters if c != c_q] or clusters
        p_rec = rng.choice(by_cluster[rng.choice(others)])

        c = parse(q_rec["a"]).contract
        issue = strip_cites(c.likely_issue)
        actions = [strip_cites(a) for a in c.actions][:3]
        if not issue or not actions:
            continue
        opener = _pick(_LIMITED_OPENERS, i)
        bridge = _pick(_LIMITED_BRIDGES, i)
        answer = render_markdown(
            issue=f"{opener} {bridge} {issue}",
            actions=actions,
            timing="",
            caution=strip_cites(c.caution)
            or sample_caution(rng, q_rec["a"], q_rec.get("meta", {}).get("cluster", "")),
            sources=None,
        )
        if "[" in answer:
            continue
        out.append({
            "id": q_rec["id"] + ":al",
            "kind": "abstain_limited",
            "q": q_rec["q"],
            "a": answer,
            # mismatched passages ARE shown at render time; nothing may be cited
            "meta": {**q_rec.get("meta", {}),
                     "passage_ids": p_rec.get("meta", {}).get("passage_ids", []),
                     "gate_passages": q_rec.get("meta", {}).get("passage_ids", []),
                     "derived": "abstain_limited"},
        })
    return out


# --------------------------------------------------------------------------
# greetings — the judge's literal first turn
# --------------------------------------------------------------------------

_GREETING_EXTRA = (
    "hey", "good morning", "good afternoon", "salaam", "bonjour",
    "hello there", "hi there", "anyone there?", "are you working?",
    "is this thing on", "test", "hello?", "yo", "greetings",
    "hi, first time using this", "hello, can you hear me",
)

_CAPABILITY_EXTRA = (
    "what do you know about?",
    "what kind of questions can I ask?",
    "how can you help me?",
    "what are you for?",
    "explain what you do",
    "who made you?",
    "what topics do you cover?",
    "can you help a farmer?",
    "what is this app?",
    "how do I use this?",
    "give me an example of something I can ask",
    "are you an expert?",
    "do you work without internet?",
    "what languages do you speak?",
)

_GREETING_OPENERS = (
    "Hi", "Hello", "Hey", "Hi there", "Hello there", "Good morning", "Good afternoon",
    "Good evening", "Salam", "As-salamu alaykum", "Hello!", "Greetings", "Morning",
    "Good day", "Hi, are you there?", "Hello, can you hear me?", "Anyone there?",
    "Hello?", "Hey there", "Hi again", "Peace be with you", "Bonjour", "Hi Sahel Sage",
    "Hello Sage",
)

_CAPABILITY_QS = (
    "What can you do?",
    "Who are you?",
    "What are you?",
    "What topics do you cover?",
    "Help",
    "What can I ask you?",
    "How can you help me?",
    "What do you know about?",
    "Can you introduce yourself?",
    "What is this?",
    "What kind of questions do you answer?",
    "Are you a farming assistant?",
    "What are you good at?",
    "Tell me what you do.",
    "How does this work?",
    "What should I ask you?",
    "Can you help me with my farm?",
    "Do you know about animals?",
    "Do you know about crops?",
    "What is your job?",
    "Explain what you are for.",
    "I am new here, what now?",
    "Give me an idea of what you can answer.",
    "Is there anything you cannot help with?",
    "What languages do you speak?",
    "Can you help an extension agent?",
    "Are you useful for a herder?",
    "What sort of advice do you give?",
    "Where does your knowledge come from?",
    "Can I ask you anything about farming?",
    "So what happens now?",
    "What do I do first?",
)

_GREET_HELLOS = (
    "Hello, and welcome.",
    "Hello!",
    "Good to hear from you.",
    "Welcome.",
    "Hello — glad you are here.",
    "Greetings, and thank you for asking.",
    "Hello.",
    "Good day.",
    "Welcome, and thank you for asking.",
    "Peace, and welcome.",
    "Hello there.",
)

_GREET_WHAT = (
    "I am Sahel Sage, an offline advisor for farmers, herders and extension agents in the Sahel and across Africa.",
    "I am Sahel Sage: I give practical farming and livestock advice, and I work entirely offline.",
    "I am Sahel Sage, a farming and livestock adviser built for smallholders working with what they have on hand.",
    "I am Sahel Sage, an agricultural assistant for smallholder farms, herds and gardens.",
    "I am Sahel Sage, and my job is practical advice on crops, animals and land.",
    "I am Sahel Sage, an offline farming and livestock advisor.",
    "I am Sahel Sage; I answer farming and herding questions without any internet.",
    "My name is Sahel Sage and I work entirely offline on this computer.",
)

_GREET_TOPICS = (
    "I can help with crops and seed choice, pests and diseases, livestock health and feeding, soil fertility, water, and storing your harvest.",
    "Ask me about planting and sowing dates, crop pests, animal health, soil, irrigation or post-harvest storage.",
    "My ground is crops, livestock, soil, water and rural livelihoods — from choosing a variety to keeping weevils out of the grain store.",
    "I cover field crops, vegetables, cattle, goats, sheep and poultry, plus soil, water and storage.",
    "Bring me questions about sowing dates, pests, animal health, soil fertility, water and grain storage.",
    "Crops, herds, soil, water and post-harvest work are all within what I can advise on.",
)


def derive_greetings(n: int = 120, rng: random.Random | None = None) -> list[dict]:
    """Short, warm, no contract headings. A judge's first message must not be
    answered with a "**Likely issue**" block."""
    questions: list[str] = list(_GREETING_OPENERS) + list(_GREETING_EXTRA)
    caps = list(_CAPABILITY_QS) + list(_CAPABILITY_EXTRA)
    for prefix in ("", "Hello. ", "Hi. ", "hey ", "Good day. "):
        questions.extend(prefix + q for q in caps)
    # case-insensitive dedup: "Hi there" and "hi there" would collapse in the
    # mixer anyway, and a silently shrinking stratum is how the scope rows got
    # down to 12 of 360 twice before.
    seen: set[str] = set()
    questions = [q for q in questions if not (q.lower() in seen or seen.add(q.lower()))]
    out = []
    for i in range(min(n, len(questions))):
        # Mixed-radix enumeration over the three pools: index i maps to a
        # DISTINCT (hello, what, topics) triple for i < 11*9*7 = 693. Ad-hoc
        # strides revisit combinations and the mixer's answer-dedup then
        # silently halves the stratum (120 of 240 in the first v5 build).
        h, rem = divmod(i, len(_GREET_WHAT) * len(_GREET_TOPICS))
        w, t = divmod(rem, len(_GREET_TOPICS))
        answer = " ".join((
            _pick(_GREET_HELLOS, h),
            _pick(_GREET_WHAT, w),
            _pick(_GREET_TOPICS, t),
        ))
        assert " not " not in f" {answer.lower()} ", (
            "greeting answers must be purely positive; refusal phrasing here is "
            "what produced round 4's 'not commercial gardening or building design'"
        )
        out.append({
            "id": f"greet:{i}",
            "kind": "greeting",
            "q": questions[i],
            "a": answer,
            "meta": {"source_docs": [], "passage_ids": [], "lang": "en",
                     "critique": "pass", "derived": "greeting"},
        })
    return out


# --------------------------------------------------------------------------
# multi-turn conversations
# --------------------------------------------------------------------------

#: (model's ONE clarifying question, the farmer's reply) — paired, and split by
#: cluster, so a poultry conversation never asks how much of the *field* is
#: affected. A mismatched clarification teaches the model that its own question
#: does not have to relate to the answer.
#: Universally applicable clarifiers — scale, season, means, urgency. These fit
#: ANY agricultural question, which is why they carry the non-symptom half of
#: the bucket.
_CLARIFY_GENERAL = (
    ("Before I answer — is this for a small plot, or across the whole farm?",
     "Just one plot for now, about half a hectare."),
    ("One question first: which season are you in, the rains or the dry season?",
     "We are at the start of the rains."),
    ("First, tell me what you have to work with — hand tools only, or equipment too?",
     "Hand tools only, and one ox plough I can borrow."),
    ("Quick question so my answer fits: is this for your own household or for selling?",
     "Mostly for selling at the weekly market."),
    ("One thing first: have you tried anything for this already?",
     "Not yet, this is the first time I am dealing with it."),
    ("Before I go further — how soon do you need to act?",
     "As soon as possible, within the next few days."),
    ("Let me check one thing: are you working alone, or do you have help for the labour?",
     "My family helps, so four of us in total."),
    ("First: can you buy inputs this season, or should I keep it to what is free?",
     "Money is tight, so keep it to what is free if you can."),
)

_CLARIFY_PLANT = (
    ("Before I answer — how much of the field is affected, a few plants or most of it?",
     "It is spreading; about half the field looks affected now."),
    ("One question first: are the plants still young, or already flowering?",
     "They are still young, only a few weeks after sowing."),
    ("Let me ask one thing first: is this a rainfed field or an irrigated one?",
     "It is rainfed, sandy soil, in the north."),
    ("First, tell me: did this start after a rain, or during the dry spell?",
     "It started right after the last rain."),
    ("Quick question so I answer the right thing — is it on the leaves, the stems or the roots?",
     "Mostly on the leaves, and now some stems too."),
    ("One thing first: has it stayed in one spot, or is it moving across the field?",
     "Only one corner so far, but it looks like it is moving."),
    ("Before I go further: how long has it been like this?",
     "About a week now, and it is getting worse."),
    ("Tell me one thing first — did you apply anything to the field recently?",
     "No, I have not applied anything this season."),
)

_CLARIFY_ANIMAL = (
    ("One question first: how many animals are showing this, and since when?",
     "Three of them, since about three days ago."),
    ("Before I answer — are they still eating and drinking normally?",
     "They are eating less than usual but still drinking."),
    ("First, tell me: are these young animals or adults?",
     "They are young ones, under a year old."),
    ("Let me ask one thing first: is the rest of the herd normal?",
     "Most of the herd is fine, it is only those few."),
    ("Quick question — has the feed or the watering point changed recently?",
     "Yes, we moved to a new watering point last week."),
    ("One thing first: is there fever, or are they cold to the touch?",
     "They feel hot and they are breathing fast."),
    ("Before I go further: have they been vaccinated this year?",
     "No, they have not been vaccinated."),
    ("Tell me one thing first — is it getting worse, or holding steady?",
     "It is getting worse each day."),
)

_REFINE_ASKS = (
    "I cannot afford that.",
    "I do not have that here.",
    "There is no agro-dealer near my village.",
    "I have no money for inputs this month.",
    "That product is not sold in our market.",
    "I have no way to buy anything right now.",
    "We do not have that equipment.",
    "That is too expensive for me.",
)

_REFINE_LEADS = (
    "Then work with what you already have.",
    "You can still make progress without buying anything.",
    "No problem — the cheap version of this still works.",
    "Leave the purchase aside and do the free part first.",
    "There is a low-cost way through this.",
    "Then start with labour rather than inputs.",
    "You can get most of the benefit at no cost.",
    "Skip the product and do these instead.",
)

_DRILL_ASKS = (
    "Explain step {n} in more detail.",
    "Can you explain the {ord} step?",
    "What exactly do you mean by step {n}?",
    "I did not understand step {n}.",
    "Say more about step {n}, please.",
    "How do I actually do step {n}?",
)

_DRILL_LEADS = (
    "Step {n} means this in practice.",
    "In detail, step {n} is:",
    "Here is step {n} broken down.",
    "Taking step {n} slowly:",
    "Step {n}, done properly:",
    "The practical version of step {n}:",
)

_DRILL_TAILS = (
    "Do it with what you already have, and check the result before moving on.",
    "Take your time with it; rushing this step is what usually makes it fail.",
    "If you can, do it early in the morning while it is still cool.",
    "Do this part yourself rather than delegating it, so you see the state of things.",
    "Repeat it once more a few days later if the change is not visible.",
    "Note the date you did it, so you know when to check again.",
)

_SWITCH_BRIDGES = (
    "Different question, ",
    "Now something else: ",
    "On another matter, ",
    "Changing subject - ",
    "Unrelated, but ",
    "New question: ",
)

_ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}


def _short_answer(c, caution: str, max_actions: int = 3, budget: int = 100) -> str:
    """Turn-2 shape: issue + a few steps + a closing caution SENTENCE.

    Two of the five headings, never all five: the system prompt promises
    follow-ups answered briefly, and re-emitting the whole block every turn is
    the repetition hook ADR-005 blames for non-termination."""
    issue = trim_words(strip_cites(c.likely_issue), 45)
    actions = [trim_words(strip_cites(a), 30) for a in c.actions][:max_actions]

    def build() -> str:
        body = render_markdown(issue=issue, actions=actions, timing="", caution="", sources=None)
        return f"{body}\n\n{caution}"

    text = build()
    while word_count(text) > budget and len(actions) > 1:
        actions.pop()
        text = build()
    return text


def _prose_answer(lead: str, steps: list[str], closing: str) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    return f"{lead}\n\n{numbered}\n\n{closing}"


def derive_multi_turn(pairs: list[dict], n: int, rng: random.Random) -> list[dict]:
    """2-3 turn conversations; only the FINAL assistant turn is trained.

    Four buckets, each roughly n/4:
      clarification — the model asks ONE targeted question, then answers;
      refinement    — "I can't afford that" -> adapted, cheaper advice;
      drill-down    — "explain step 2" -> that step only, short;
      topic switch  — an unrelated new question, with no stale context.

    Turn-2 answers are deliberately short (60-120 words) and never repeat the
    full five-heading structure: the system prompt promises exactly that, and
    a model that re-emits the whole block on every follow-up is the repetition
    hook ADR-005 blames for the non-termination bug.
    """
    usable = [
        rec for rec in pairs
        if len(parse(rec["a"]).contract.actions) >= 2 and parse(rec["a"]).contract.likely_issue
    ]
    if len(usable) < 8:
        return []
    per = max(1, n // 4)
    out: list[dict] = []

    def conv(rec_id: str, bucket: str, turns: list[dict], meta: dict) -> dict:
        return {
            "id": f"{rec_id}:mt:{bucket}",
            "kind": "multi_turn",
            "turns": turns,
            "meta": {**meta, "passage_ids": [],
                     "gate_passages": meta.get("passage_ids", []), "critique": "pass",
                     "derived": "multi_turn", "bucket": bucket},
        }

    # (a) clarification -----------------------------------------------------
    for i, rec in enumerate(rng.sample(usable, min(per, len(usable)))):
        c = parse(rec["a"]).contract
        answer = _short_answer(c, sample_caution(rng, rec["a"], cluster(rec)))
        if looks_like_a_problem(rec["q"]):
            pool = _CLARIFY_ANIMAL if cluster(rec) == "livestock" else _CLARIFY_PLANT
        else:
            pool = _CLARIFY_GENERAL
        ask, reply = _pick(pool, i)
        turns = [
            {"q": rec["q"], "a": ask},
            {"q": reply, "a": answer},
        ]
        out.append(conv(rec["id"], "clarification", turns, rec.get("meta", {})))

    # (b) refinement --------------------------------------------------------
    for i, rec in enumerate(rng.sample(usable, min(per, len(usable)))):
        c = parse(rec["a"]).contract
        fallback = [trim_words(strip_cites(a), 28) for a in c.actions[1:]][:3]
        if not fallback:
            fallback = [trim_words(strip_cites(c.actions[0]), 28)]
        answer = _prose_answer(
            _pick(_REFINE_LEADS, i), fallback, sample_caution(rng, rec["a"], cluster(rec))
        )
        turns = [
            {"q": rec["q"], "a": uncite(rec["a"])},
            {"q": _pick(_REFINE_ASKS, i), "a": answer},
        ]
        out.append(conv(rec["id"], "refinement", turns, rec.get("meta", {})))

    # (c) drill-down --------------------------------------------------------
    for i, rec in enumerate(rng.sample(usable, min(per, len(usable)))):
        c = parse(rec["a"]).contract
        step = (i % min(len(c.actions), 3)) + 1
        ask = _pick(_DRILL_ASKS, i).format(n=step, ord=_ORDINALS[step])
        body = trim_words(strip_cites(c.actions[step - 1]), 45)
        answer = " ".join((
            _pick(_DRILL_LEADS, i).format(n=step),
            body if body.endswith((".", "!", "?")) else body + ".",
            _pick(_DRILL_TAILS, i),
        ))
        turns = [
            {"q": rec["q"], "a": uncite(rec["a"])},
            {"q": ask, "a": answer},
        ]
        out.append(conv(rec["id"], "drill_down", turns, rec.get("meta", {})))

    # (d) topic switch ------------------------------------------------------
    firsts = rng.sample(usable, min(per, len(usable)))
    seconds = rng.sample(usable, min(per, len(usable)))
    for i, (first, second) in enumerate(zip(firsts, seconds, strict=False)):
        if first["id"] == second["id"]:
            continue
        c = parse(second["a"]).contract
        answer = _short_answer(c, sample_caution(rng, second["a"], cluster(second)))
        turns = [
            {"q": first["q"], "a": uncite(first["a"])},
            {"q": _pick(_SWITCH_BRIDGES, i) + second["q"][0].lower() + second["q"][1:],
             "a": answer},
        ]
        out.append(conv(f"{first['id']}|{second['id']}", "topic_switch", turns,
                        second.get("meta", {})))

    return out


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------


def derive_all(
    pairs_path: Path,
    out_path: Path,
    n_closed: int = 1800,
    n_bare: int = 1800,
    n_limited: int = 1000,
    n_scope: int = 450,
    n_greetings: int = 120,
    n_multi_turn: int = 1000,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]

    # closed_book / bare / multi_turn draw from DISJOINT pools: the same pair
    # rendered twice with the same completion is duplicated supervision, not
    # extra coverage.
    order = list(range(len(pairs)))
    rng.shuffle(order)
    cut1, cut2 = n_closed, n_closed + n_bare
    pool_closed = [pairs[i] for i in order[:cut1]]
    pool_bare = [pairs[i] for i in order[cut1:cut2]]
    pool_multi = [pairs[i] for i in order[cut2:]] or pairs

    strata = {
        "closed_book": derive_closed_book(pool_closed, n_closed, rng),
        "bare": derive_bare_questions(pool_bare, n_bare, rng),
        "abstain_limited": derive_abstain_limited(pairs, n_limited, rng),
        "abstain_scope": derive_abstain_scope(n_scope, rng),
        "greeting": derive_greetings(n_greetings, rng),
        "multi_turn": derive_multi_turn(pool_multi, n_multi_turn, rng),
    }
    derived = [rec for rows in strata.values() for rec in rows]
    rng.shuffle(derived)
    with out_path.open("w") as f:
        for rec in derived:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    counts = {k: len(v) for k, v in strata.items()}
    counts["total"] = len(derived)
    return counts


__all__ = [
    "CAUTION_VARIANTS",
    "SCOPE_ITEMS",
    "derive_abstain_limited",
    "derive_abstain_scope",
    "derive_all",
    "derive_bare_questions",
    "derive_closed_book",
    "derive_greetings",
    "derive_multi_turn",
    "scope_answer",
    "strip_cites",
    "trim_words",
    "uncite",
]
