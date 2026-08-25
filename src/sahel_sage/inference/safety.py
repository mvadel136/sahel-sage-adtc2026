"""Hard safety layer: refuse before the model runs, and check what it wrote.

The expert audit of 2026-08-13 found five dangerous answers in twenty-four. All
five came from questions whose *class* is unsafe for a generative model to
answer at all — a pesticide mixing rate, a pre-harvest interval, an antibiotic
decision, a human drug given to a cow, a pesticide container reused for drinking
water, a twelve-year-old sent to spray. Every one of those was already forbidden
in the system prompt (`core.prompts._BASE`), and the model overrode the prompt
in every one of those cases.

So prohibitions stop being instructions and become code. Two gates:

**Pre-generation.** `screen()` matches the question against a curated set of
absolute prohibitions and returns a fixed, human-written answer. The model is
never consulted. This is the gate that makes the five dangerous outputs
impossible rather than unlikely — a guarantee, not a probability.

**Post-generation.** `unsupported_quantities()` (imported from the training-time
gate, which has been applied to datasets since v5) is re-applied to what the
model actually wrote: any measured quantity absent from the retrieved passages
means the answer is rejected. A number the sources do not contain is a number
the model invented.

Design notes:

* Qualifiers must not move the answer. "if I wash it first", "just this once",
  "he's careful" are exactly how a farmer asks these questions, and a model that
  weighs them will eventually say yes. The patterns therefore match the *topic*,
  not the request, and there is no path from a matched prohibition to a
  generated answer.
* Refusals are written to be useful. "I can't help with that" teaches a farmer
  to stop asking; naming who *can* answer — the label, the dealer, the extension
  agent, the vet — is the part that has value offline.
* Scope follows the Plantwise green/yellow split (CABI Pest Management Decision
  Guides): prevention, monitoring, physical and biological control are safe to
  advise anywhere; chemical control is jurisdiction-specific and is refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from sahel_sage.training.numguard import unsupported_quantities

__all__ = [
    "PROHIBITIONS",
    "Prohibition",
    "Refusal",
    "RefusalReason",
    "screen",
    "unsupported_quantities",
]


class RefusalReason(StrEnum):
    """Why we did not answer.

    The three `OUT_OF_*` values are Digital Green's Farmer.Chat taxonomy, which
    they report as a headline metric rather than hiding: a refusal that says
    *which kind* of gap it hit tells us whether to add documents, redirect the
    user, or widen the crop list.
    """

    #: An absolute prohibition. Never answerable, by us or by a larger model.
    PROHIBITED = "prohibited"
    #: In scope and on topic, but the library has nothing on it.
    OUT_OF_CONTENT = "out_of_content"
    #: Not an agriculture question at all.
    OUT_OF_CONTEXT = "out_of_context"
    #: Agriculture, but a crop, animal or region we do not cover.
    OUT_OF_COLLECTION = "out_of_collection"


@dataclass(frozen=True)
class Refusal:
    reason: RefusalReason
    #: The exact text shown to the user. Human-written, never generated.
    text: str
    #: Which prohibition fired, for logging and for the audit. Empty for the
    #: retrieval-gap reasons.
    rule: str = ""


@dataclass(frozen=True)
class Prohibition:
    id: str
    #: Matched against the lowercased question. Any hit fires the rule.
    patterns: tuple[re.Pattern[str], ...]
    response: str
    #: Why this is absolute, in one line — quoted in REPORT.md and in the
    #: adversarial test suite so the rationale cannot drift from the rule.
    rationale: str


#: The identical opening of every refusal.
#:
#: Safety behaviour is concentrated in a model's first few output tokens
#: (arXiv:2406.05946, "Safety Alignment Should Be Made More Than Just a Few
#: Tokens Deep"). Seven differently-worded refusals therefore taught seven
#: separate weak transitions out of the question and into a decline. One shared
#: prefix teaches a single strong one, and the specific reasoning follows it
#: unchanged — the farmer still gets a different answer for each topic.
#:
#: Kept short deliberately: it is the token sequence doing the work, so it has
#: to be reachable from any prohibited question, not tuned to one of them.
REFUSAL_OPENING = "I can't answer that one, and the reason matters."


def _refusal(issue: str, actions: list[str], timing: str, caution: str) -> str:
    """Assemble a refusal in the answer contract, behind the shared opening.

    Written as a function rather than seven hand-formatted strings so the
    contract shape and the opening cannot drift apart between rules — which is
    exactly how the prose-versus-contract mismatch got into round 6.
    """
    steps = "\n".join(f"{i}. {a}" for i, a in enumerate(actions, 1))
    return (
        f"**Likely issue**\n{REFUSAL_OPENING} {issue}\n\n"
        f"**What to do**\n{steps}\n\n"
        f"**Timing**\n{timing}\n\n"
        f"**Caution**\n{caution}"
    )


def _p(*sources: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(s, re.IGNORECASE) for s in sources)


# The `[\s\S]{0,N}` filler between the two halves of a two-part pattern
# crosses sentence boundaries on purpose: farmers state the situation and then
# ask, in that order — "My son is 12. Can he spray the field?" — and a first
# version of these rules used `[^?.]`, which missed three of the six questions
# the expert audit had flagged as dangerous. Bounded by length instead.


#: Words that mean "a plant-protection or veterinary chemical" to a farmer.
_CHEMICAL = (
    r"(?:pesticid\w*|insecticid\w*|herbicid\w*|fungicid\w*|acaricid\w*"
    r"|weedkiller|spray|chemical|product|dose|dosage|drug|medicine|treatment)"
)

#: Active ingredients a Sahelian farmer is likely to name directly. These are
#: mostly legal, registered products — they are here so that a *rate* question
#: is caught when the farmer names the chemical instead of the word "pesticide".
#: Banned products live in their own rule below and are refused outright.
_ACTIVE_INGREDIENT = (
    r"(?:deltamethrin|cypermethrin|lambda[- ]?cyhalothrin|cyhalothrin|permethrin"
    r"|imidacloprid|acetamiprid|thiamethoxam|emamectin|abamectin|spinosad"
    r"|chlorpyrifos|profenofos|malathion|dimethoate|pirimiphos[- ]?methyl"
    r"|mancozeb|metalaxyl|thiram|copper oxychloride|sulphur dust|sulfur dust"
    r"|glyphosate|atrazine|pendimethalin|2,?4[- ]?d\b|nicosulfuron)"
)

#: "how much / how many / what dose / what is the right dose / how strong".
#: The `(?:is|are) the ...` arm was added after the prohibition training-data
#: generator asserted its own questions against this pattern and found that
#: "what is the right dose of insecticide for my sprayer?" walked straight
#: through, while the terser "what dose of insecticide" was caught.
_QUANTITY_WORD = r"(?:rate|dose|dosage|amount|quantity|concentration|mix|mixture|strength)"
_HOW_MUCH = (
    r"(?:how (?:much|many)|how strong"
    rf"|what (?:{_QUANTITY_WORD}"
    rf"|(?:is|are|was) the (?:right |correct |proper |best |recommended |usual |normal )?{_QUANTITY_WORD}"
    r"))"
)

#: The equipment a rate question is asked in terms of. "How much X in a 15 litre
#: knapsack?" names no chemical class at all. Deliberately excludes bucket,
#: drum, tank and watering can — "how much water in the watering can for my
#: seedlings" is a perfectly good question and must not be refused.
_SPRAY_KIT = r"(?:knapsack|back ?pack sprayer|sprayer|spray tank|spray pump)"


PROHIBITIONS: tuple[Prohibition, ...] = (
    Prohibition(
        id="pesticide_rate",
        rationale=(
            "Mixing rates are product-specific and set by national registration; "
            "a wrong rate poisons the applicator, the crop or the consumer."
        ),
        patterns=_p(
            rf"{_HOW_MUCH}\b[\s\S]{{0,60}}\b{_CHEMICAL}",
            rf"{_CHEMICAL}[\s\S]{{0,40}}\bper (?:litre|liter|l\b|hectare|ha\b|knapsack|tank|drum)",
            r"\b(?:ml|millilitres?|milliliters?|grams?|g)\b[\s\S]{0,30}\bper (?:litre|liter|l\b)",
            r"\bmix(?:ing)?\b[\s\S]{0,40}\bper (?:litre|liter|l\b|hectare|ha\b)",
            r"\b(?:dilut\w+|mixing) (?:rate|ratio)\b",
            rf"{_HOW_MUCH}\b[\s\S]{{0,60}}\b{_ACTIVE_INGREDIENT}",
            rf"\b{_ACTIVE_INGREDIENT}\b[\s\S]{{0,60}}\b{_SPRAY_KIT}",
            rf"{_HOW_MUCH}\b[\s\S]{{0,60}}\b{_SPRAY_KIT}",
            # Asking to CONFIRM or ADJUST a rate is asking for a rate. Writing
            # only "how much X" caught the textbook phrasing and missed every
            # way a farmer actually raises it — with a number already in hand,
            # or by asking to go stronger.
            r"\b(?:is|are)\b[\s\S]{0,30}\b(?:spoon|cap|capful|lid|handful|scoop|"
            r"sachet|glass)\b[\s\S]{0,40}\bper\b",
            r"\b(?:two|three|one|a|\d+)\s*(?:caps?|spoons?|lids?|scoops?)\b"
            r"[\s\S]{0,40}\b(?:per|sprayer|knapsack|tank)\b",
            r"\bmake the (?:spray|mix|solution|dose)\b[\s\S]{0,30}\bstronger\b",
            r"\bstronger\b[\s\S]{0,40}\b(?:spray|mix|dose|solution)\b",
            r"\bwork out the (?:amount|quantity|dose|rate)\b",
            r"\bcannot read (?:it|the label|the bag)\b[\s\S]{0,60}\bhow much\b",
        ),
        response=_refusal(
            issue='I will not give you a number, and I will not invent an example one. No one can give you a mixing rate without seeing your container: the correct rate differs for every product, crop and pest, and is set by whoever registered that product in your country.',
            actions=[
                'The only trustworthy rate is the one printed on the label of your own container, for your crop and your pest.',
                'If you cannot read the label, take the container itself to your agro-dealer or extension agent and have them read it to you. Do not describe it from memory — take the container.',
                'If the container has no label, do not use it.',
                'Tell me the crop and what you are seeing, and I will help with prevention, monitoring and non-chemical control instead.',
            ],
            timing='Read the label before you mix anything.',
            caution='Too little wastes your money. Too much poisons you, your soil and whoever eats the crop.',
        ),
    ),
    Prohibition(
        id="preharvest_interval",
        rationale=(
            "The pre-harvest interval is product-specific and legally binding; "
            "harvesting inside it puts residues above the Codex MRL on food sold "
            "to other people."
        ),
        patterns=_p(
            r"\b(?:spray\w*|treat\w*|applied)\b[\s\S]{0,80}\b(?:harvest|pick|sell|market|eat)",
            r"\b(?:harvest|pick|sell|market|eat)\b[\s\S]{0,80}\b(?:spray\w*|treat\w*|pesticid\w*)",
            r"\bpre-?harvest interval\b|\bwithholding period\b|\bhow long (?:before|after)\b[\s\S]{0,40}\bspray",
            # A spray date plus an intention to sell or eat. Farmers rarely use
            # the words "pre-harvest interval"; they say "the buyer comes this
            # afternoon and I sprayed Monday".
            r"\b(?:buyer|market|customer)\b[\s\S]{0,80}\bspray\w*\b",
            r"\bspray\w*\b[\s\S]{0,80}\b(?:buyer|market|customer)\b",
            r"\btreated\b[\s\S]{0,60}\b(?:recently|this week|yesterday|today)\b",
            r"\b(?:family|children|we)\b[\s\S]{0,40}\beat\b[\s\S]{0,60}\bspray\w*\b",
        ),
        response=_refusal(
            issue='Do not harvest or sell it yet. Every pesticide has a pre-harvest interval — the days that must pass between the last spray and harvest. It is printed on the label and differs for every product.',
            actions=[
                'Find the container you sprayed from and read the pre-harvest interval on the label.',
                'Count that many days from the day you sprayed, not from today.',
                'If you cannot find the label or cannot read it, take the container to your extension agent or agro-dealer before you harvest.',
                'If you have already sold some, tell your buyer. That is a much smaller problem today than it will be later.',
            ],
            timing='Wait out the full interval from the day of spraying.',
            caution='Selling inside that window puts chemical residues on food other people will eat, and in most countries it is also illegal.',
        ),
    ),
    Prohibition(
        id="veterinary_drug",
        rationale=(
            "Antibiotic choice, dose and withdrawal period require a diagnosis "
            "and a prescription; wrong use kills the animal, contaminates milk "
            "and meat, and drives resistance."
        ),
        patterns=_p(
            r"\b(?:antibiotic|antibiotics|oxytetracycline|tetracycline|penicillin"
            r"|amoxicillin|ivermectin|albendazole|dewormer|anthelmintic|vaccine dose)\b",
            rf"{_HOW_MUCH}\b[\s\S]{{0,60}}\b(?:inject|injection|tablet|bolus|syringe)",
            r"\bwithdrawal period\b",
            # Choosing the product is the same decision as choosing the dose,
            # and it is how the question actually arrives: standing in the
            # animal pharmacy looking at bottles.
            r"\bwhich (?:one|bottle|product|drug|medicine|antibiotic)\b"
            r"[\s\S]{0,60}\b(?:goat|sheep|cow|calf|camel|chicken|animal)\b",
            r"\bwhat (?:injection|treatment|medicine|drug)\b[\s\S]{0,60}"
            r"\b(?:goat|sheep|cow|calf|camel|chicken|animal)\b",
            r"\b(?:goat|sheep|cow|calf|camel|chicken|animal)\b[\s\S]{0,60}"
            r"\bwhat (?:injection|treatment|medicine|drug)\b",
            r"\banimal pharmacy\b|\bveterinary pharmacy\b",
        ),
        response=_refusal(
            issue='Choosing an antibiotic or a dewormer needs a diagnosis first: the same signs come from very different diseases, and the wrong drug wastes the treatment while the animal keeps getting worse.',
            actions=[
                'Contact your veterinarian or veterinary auxiliary. Describe what you see and how many animals are affected.',
                'Separate sick animals from the rest of the herd now. That part does not need a diagnosis and it limits the spread.',
                'Keep clean water available. Most losses in scouring animals are from dehydration, not from the infection itself.',
                'Tell me the signs you are seeing and I will help you describe them accurately to the vet.',
            ],
            timing='Separate the animals today. Get the diagnosis from a vet before any medicine is given.',
            caution="The dose depends on the animal's weight and the exact product, and after treatment there is a withdrawal period during which the milk and meat must not be sold.",
        ),
    ),
    Prohibition(
        id="human_medicine_for_animals",
        rationale=(
            "Human formulations are dosed for humans and several are toxic to "
            "livestock; residues also enter milk and meat with no withdrawal data."
        ),
        patterns=_p(
            r"\b(?:paracetamol|acetaminophen|ibuprofen|aspirin|diclofenac"
            r"|amoxicillin capsules?|human (?:medicine|drug|tablet|pill|dose))\b",
            r"\b(?:medicine|tablet|pill|drug)s?\b[\s\S]{0,40}\bfrom the (?:pharmacy|clinic|chemist)\b",
            # "my cow is in pain, can I use my own medicine?" — a farmer naming
            # no drug at all. Found by the training generator, not by us.
            r"\b(?:my|our|your) own (?:medicine|tablets?|pills?|drugs?)\b",
            r"\b(?:medicine|tablets?|pills?|drugs?) (?:i|we) (?:take|use|have)\b",
            # The farmer rarely says "human medicine". They say "the tablets
            # from the clinic" or "half a tablet", and the animal is named
            # nearby.
            r"\b(?:half a |a |some )?tablets?\b[\s\S]{0,60}"
            r"\b(?:goat|sheep|cow|calf|camel|chicken|animal)\b",
            r"\b(?:goat|sheep|cow|calf|camel|chicken|animal)\b[\s\S]{0,60}"
            r"\b(?:half a |a |some )?tablets?\b",
            r"\bmedicine (?:at home|from the clinic|from the pharmacy)\b",
            r"\b(?:clinic|pharmacy|chemist)\b[\s\S]{0,70}\b(?:only sells for people"
            r"|for people|for humans)\b",
            r"\bgive (?:my |the )?(?:cow|goat|sheep|camel|chicken|animal)s?\b[\s\S]{0,40}"
            r"\b(?:my|human|our) (?:medicine|tablet|pill|drug)",
        ),
        response=_refusal(
            issue='Do not give human medicines to livestock. Human tablets are measured for a human body, and several are actively poisonous to farm animals even in small amounts.',
            actions=[
                'Call your veterinarian or veterinary auxiliary and describe the signs.',
                'Move the animal into shade, keep clean water in front of it, and separate it from the others.',
                'Note when the signs started and whether any other animals show them. That is the information the vet needs most.',
                'If no vet can reach you, tell me what you are seeing and I will help with the care that does not need a drug.',
            ],
            timing='Move and water the animal now; call the vet today.',
            caution='Damage to the liver, kidneys or gut often shows only when it is too late to reverse, and there is no safe waiting time established for milk or meat afterwards — the residues reach whoever drinks or eats from that animal.',
        ),
    ),
    Prohibition(
        id="pesticide_container_reuse",
        rationale=(
            "Pesticide residues bind to the plastic and cannot be removed by "
            "washing; FAO/WHO guidance is triple-rinse, puncture, and never reuse."
        ),
        patterns=_p(
            # A NARROWER chemical list than _CHEMICAL here. That one includes
            # "treatment", "product" and "spray" so it can catch "what treatment
            # dose" — useful for a rate question, wrong for naming a vessel.
            # "cold treatment container" for strawberries matched it and refused
            # a perfectly good storage question.
            r"\b(?:empty |used |old )?"
            r"(?:pesticid\w*|insecticid\w*|herbicid\w*|fungicid\w*|acaricid\w*"
            r"|weedkiller|agrochemical|chemical) "
            r"(?:container|bottle|can|drum|jerr?y ?can|tin)\b",
            # The vessel needs a reuse signal — chemical, empty, or washed —
            # before this fires. Without one it matched "I have a pallet of
            # strawberries ... what cold treatment ... store", refusing an
            # ordinary storage question as container reuse. Over-refusal is not
            # a safe failure: Bianchi et al. measured exaggerated safety making
            # models decline perfectly good questions, and a judge refused a
            # fair question scores that as badly as a wrong answer.
            #
            # "washed/rinsed" earns its place here: "I washed the drum very
            # well, is it fine for drinking water now?" names no chemical at
            # all, and requiring one lost it. Nobody washes a drum before
            # storing strawberries in it.
            rf"\b(?:{_CHEMICAL}|empty|wash\w*|rins\w*|clean\w*)\b[\s\S]{{0,40}}"
            r"\b(?:container|bottle|drum|jerr?y ?can|tin)\b[\s\S]{0,60}"
            r"\b(?:drinking water|water|food|grain|seed|milk|carry|reuse|re-?use)\b",
            # Disposal questions land here too — "what do I do with the empty
            # drums?" — and that is correct: our answer IS the disposal advice
            # (triple-rinse, puncture, collection point). Refusing to improvise
            # and giving the FAO procedure are the same response.
            r"\bempty (?:drums?|containers?|cans?|bottles?)\b",
            r"\bwhat (?:should I |can I |to )?do with the (?:drums?|containers?|"
            r"cans?|bottles?)\b",
            r"\bboiled? the (?:container|drum|can|bottle)\b",
            r"\b(?:keep|store|put)\b[\s\S]{0,30}\bin the (?:empty )?"
            r"(?:spray |chemical |pesticide )?(?:drum|container|can|jerr?y ?can)\b",
        ),
        response=_refusal(
            issue='Washing it does not make it safe. Pesticide soaks into the plastic itself. Rinsing removes what you can see, not what has gone into the walls of the container, and it keeps leaching out into whatever you put in next.',
            actions=[
                'Rinse the empty container three times and pour each rinse into the spray tank. That liquid still works as spray and it is where the last of the product belongs.',
                'Puncture or crush the container so nobody can reuse it, including you.',
                'Ask your extension agent or agro-dealer whether there is a collection point for empty containers in your area. If there is none, bury it away from wells, watercourses and where animals graze.',
                'Carry drinking water in a container that has only ever held food or water.',
            ],
            timing='Rinse and puncture the empty container as soon as it is empty. It never becomes a water container, at any point.',
            caution='Water, milk, grain and cooking oil all pick up the residue. Children are poisoned this way every year, and it is one of the most common causes of pesticide poisoning in farming households. Never store food or water in it, and never let it become a toy.',
        ),
    ),
    Prohibition(
        id="minor_applying_pesticide",
        rationale=(
            "ILO Convention 182 lists pesticide application as hazardous child "
            "labour; children absorb more per kilo of body weight than adults."
        ),
        patterns=_p(
            # "young" is deliberately absent: "my young maize needs spraying" is
            # a legitimate question and must not be refused as child labour.
            r"\b(?:child|children|kid|boy|girl|son|daughter|minor|teenager)\b"
            r"[\s\S]{0,80}\b(?:spray|apply|mix|pesticid\w*|insecticid\w*|herbicid\w*|chemical)",
            r"\b(?:spray|apply|mix)\w*\b[\s\S]{0,60}"
            r"\b(?:my son|my daughter|my child|the children|a child|the boy|the girl)\b",
            r"\b(?:aged?|years? old)\b[\s\S]{0,40}\bspray",
            # Families do not say "minor". They say "my nephew is fifteen" or
            # "the boys mix it", and the age is stated as a word.
            r"\b(?:nephew|niece|grandson|granddaughter|the boys|the girls|"
            r"my sons?|my daughters?|schoolboy)\b[\s\S]{0,80}"
            r"\b(?:spray|sprayer|mix|chemical|pesticid\w*|field work)\b",
            r"\b(?:spray|sprayer|mix|chemical|pesticid\w*)\b[\s\S]{0,80}"
            r"\b(?:nephew|niece|grandson|the boys|the girls|my sons?|my daughters?)\b",
            r"\b(?:is|aged?|only|just)\s+(?:ten|eleven|twelve|thirteen|fourteen|"
            r"fifteen|sixteen|seventeen)\b[\s\S]{0,80}"
            r"\b(?:spray|sprayer|mix|chemical|carry)\b",
            r"\bat what age\b[\s\S]{0,50}\b(?:spray|chemical|pesticid\w*)\b",
            r"\b(?:my sons?|my daughters?|the children|the kids)\b[\s\S]{0,60}"
            r"\b(?:help|work)\b[\s\S]{0,30}\b(?:in the field|on the farm)\b",
            r"\bwho (?:in the family |else )?(?:is allowed|can|should)\b"
            r"[\s\S]{0,50}\b(?:handle|spray|mix)\b[\s\S]{0,30}\bchemical",
            # A stated age under 18 with no child noun: "He is only 14 but he
            # has done it before. Can he mix the chemical?"
            r"\b(?:is|aged?|only|just|he'?s|she'?s)\s+(?:1[0-7]|[1-9])\b"
            r"[\s\S]{0,80}\b(?:spray|apply|mix|pesticid\w*|insecticid\w*|herbicid\w*|chemical)",
        ),
        response=_refusal(
            issue="No one under 18 may spray, mix or carry pesticides — not a child, not a strong teenager of fifteen or sixteen — and this does not depend on how careful they are. A young body absorbs more chemical for its size than an adult's, and the harm falls on organs and nerves that are still developing.",
            actions=[
                'Let the spraying wait. Almost no pest problem gets seriously worse in the few days it takes for an adult to be available.',
                'Ask another adult, or hire one for the day.',
                "Keep children out of the field for as long as the label's re-entry period says after any spraying.",
                'Give the child other work. There is plenty; this is not it.',
            ],
            timing='Wait for an adult, however long that takes.',
            caution='Protective equipment is made for adult bodies and does not seal on a child. Pesticide application is classified internationally as hazardous work that no one under 18 should do — including mixing, loading, spraying and washing the equipment afterwards.',
        ),
    ),
    Prohibition(
        id="banned_pesticide",
        rationale=(
            "WHO Class Ia/Ib and Rotterdam Convention PIC products; several are "
            "banned outright across the Sahel and none is safe to advise on."
        ),
        patterns=_p(
            r"\b(?:paraquat|paraqu?at|aldicarb|monocrotophos|methyl parathion"
            r"|parathion|endosulfan|dichlorvos|ddt|lindane|carbofuran|phorate"
            r"|methamidophos|dieldrin|aldrin|chlordane|heptachlor|dibromide"
            r"|phosphine tablets?|aluminium phosphide|aluminum phosphide)\b",
            # Nobody asks for a WHO Class Ia product by name. They ask for the
            # strongest thing in the market, and the informal market answers.
            r"\bkills? everything\b",
            r"\b(?:strongest|most powerful|hardest|toughest)\b[\s\S]{0,50}"
            r"\b(?:chemical|pesticid\w*|product|powder|spray|weedkiller|thing)\b",
            r"\bwhat is the strongest\b",
            r"\bwhich chemical kills\b[\s\S]{0,30}\bfastest\b",
            r"\bcheap (?:powder|chemical|pesticide)\b[\s\S]{0,40}\bkills?\b",
        ),
        response=_refusal(
            issue='That product belongs to a group of pesticides banned or severely restricted in most countries, including across the Sahel, because they cause severe poisoning at small doses and there is often no antidote.',
            actions=[
                'Do not use it. If you have some, do not pour it onto the ground or into water. Ask your extension service how to dispose of it.',
                'Keep it locked away from children and animals until it can be disposed of.',
                'Tell your extension agent or agro-dealer what the pest problem actually is. There is a registered product or a non-chemical method for almost all of them.',
                'Tell me the crop and the pest and I will go through prevention and non-chemical control with you.',
            ],
            timing='Stop using it now; arrange disposal when you can.',
            caution='Some of these are still sold informally in markets. That does not make them legal or safe.',
        ),
    ),
)


def screen(question: str) -> Refusal | None:
    """Match a question against the absolute prohibitions.

    Returns the fixed response to send instead of generating, or None when the
    question may proceed to the model. Order matters only in that the first
    match wins; the rules are written to be near-disjoint, and where they do
    overlap (a dose of a banned product) either answer is correct.
    """
    if not question or not question.strip():
        return None
    for rule in PROHIBITIONS:
        if any(p.search(question) for p in rule.patterns):
            return Refusal(reason=RefusalReason.PROHIBITED, text=rule.response, rule=rule.id)
    return None
