"""Counter-prior training data: beating the base model's temperate assumptions.

Round 4 said millet is "a cool-season crop, plant in early spring or late
autumn" and treated Newcastle disease with "sanitation". Neither is a knowledge
gap — both are Qwen3's Northern-Hemisphere pretraining prior surviving our
fine-tune.

The mechanism is documented and it is about *magnitude*, not correctness: an
adapter's margin stays roughly constant while the pretrained margin grows with
how often the model saw the prior (arXiv 2604.23750, "The Override Gap"), and
fine-tuning gradients disproportionately reinforce pre-existing parametric
priors (arXiv 2410.10796). Geographic bias against Sub-Saharan contexts
specifically is measured in arXiv 2402.02680.

Two consequences shape this module:

1. **Oversample.** A counter-prior fact needs materially more exposures than a
   prior-consistent one, so each conflict expands into several phrasings.
2. **Name the wrong answer.** Stating the correct fact is not enough when a
   competing fact is already strongly encoded; the correction must be
   contrastive — "not X; Z" — so the wrong association is explicitly suppressed.

Every entry carries the corpus statement that justifies it, exactly like
`evaluation/factcheck.py`. Ground truth is FAO/ICRISAT/WOCAT text, not ours.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from sahel_sage.training.markdownify import render_markdown
from sahel_sage.training.normalize import sample_caution


@dataclass(frozen=True)
class PriorConflict:
    name: str
    #: several ways a user might ask
    questions: tuple[str, ...]
    #: the temperate/incorrect belief, named explicitly so it is suppressed
    wrong: str
    #: the correct statement for the Sahel
    right: str
    actions: tuple[str, ...]
    timing: str
    source: str
    cluster: str = "crops"


CONFLICTS: tuple[PriorConflict, ...] = (
    PriorConflict(
        name="millet_sowing",
        questions=(
            "When should I plant millet?",
            "when do i sow millet",
            "What is the right planting time for pearl millet in the Sahel?",
            "My neighbour says to plant millet in spring. Is that right?",
            "when is the best month to plant millet in mauritania",
        ),
        wrong="Millet is not a cool-season or spring-sown crop here, and it is not planted "
              "in the dry season.",
        right="In the Sahel millet is rainfed: it is sown at the onset of the rainy season, "
              "once the first good rains have wet the soil to planting depth.",
        actions=(
            "Wait for the first rains that wet the soil properly rather than sowing on dust.",
            "Sow within a few days of that rain so the crop uses the whole wet season.",
            "Keep seed ready before the rains so you are not delayed when they arrive.",
        ),
        timing="At the onset of the rains, not on a fixed calendar month.",
        source="wocat-rangeland-ssa: 'sowed immediately after the onset of the rainy season'",
    ),
    PriorConflict(
        name="newcastle_control",
        questions=(
            "My village chickens keep dying in outbreaks. What prevents it?",
            "chickens dying suddenly in my village what do i do",
            "How do I stop Newcastle disease in my flock?",
            "Is cleaning the coop enough to stop chickens dying?",
        ),
        wrong="Cleaning and disinfecting the housing alone will not stop Newcastle disease, "
              "and antibiotics do not cure it because it is a virus.",
        right="Newcastle disease is controlled by vaccination. Village flocks need a "
              "vaccination programme repeated on schedule, alongside good hygiene.",
        actions=(
            "Ask your veterinary service or extension agent about the Newcastle vaccine "
            "used for village poultry in your area.",
            "Vaccinate the whole flock and repeat according to the schedule you are given.",
            "Separate new or sick birds from the flock and do not move birds during an outbreak.",
        ),
        timing="Before the season when outbreaks usually occur, then on the repeat schedule.",
        source="fao-newcastle-village-chickens-field-manual: vaccination is the control measure",
        cluster="livestock",
    ),
    PriorConflict(
        name="season_naming",
        questions=(
            "What should I do on my farm in winter?",
            "what crops grow in summer here",
            "How do I plan my farming year?",
        ),
        wrong="The farming year here is not organised into spring, summer, autumn and winter.",
        right="It runs on the rainy season and the dry season. The rains set planting, "
              "weeding and harvest; the dry season is for land preparation, storage, "
              "irrigation and livestock feeding.",
        actions=(
            "Plan field operations around the start and end of the rains, not calendar seasons.",
            "Use the dry season for repairs, manure application and preparing storage.",
            "Keep fodder and water plans for the late dry season, when feed is scarcest.",
        ),
        timing="Plan before the rains begin.",
        source="agro-pastoral calendars throughout the Sahel cluster documents",
        cluster="sahel",
    ),
    PriorConflict(
        name="grain_storage",
        questions=(
            "How do I store my grain so insects do not eat it?",
            "weevils in my stored maize what do i do",
            "Do I need a big silo to store my harvest safely?",
        ),
        wrong="You do not need a mechanised silo or a cold store, and grain must not be "
              "put into storage while it is still damp.",
        right="Dry the grain properly first, then store it sealed from air. Airtight "
              "storage such as hermetic bags kills storage insects without chemicals.",
        actions=(
            "Dry the grain until it is hard and does not dent when pressed.",
            "Clean the store and remove all old grain before putting the new harvest in.",
            "Fill hermetic bags completely and close them tightly so no air gets in.",
        ),
        timing="Dry and store immediately after harvest, before insects establish.",
        source="purdue-pics-bags-grain-storage-guide: dry grain, then airtight storage",
        cluster="pest",
    ),
    PriorConflict(
        name="saline_soil",
        questions=(
            "There is white salt crust on my irrigated plot. What can I do?",
            "my soil is salty how do i fix it",
            "Should I just add more fertilizer to salty soil?",
        ),
        wrong="Adding more fertiliser does not fix salinity, and simply stopping irrigation "
              "usually makes it worse because salts stay in the root zone.",
        right="Salts have to be washed below the roots and then drained away. That needs "
              "extra irrigation water for leaching plus somewhere for the water to go.",
        actions=(
            "Make sure drainage works before leaching, or the water table will rise.",
            "Apply extra water beyond crop needs to wash salts below the root zone.",
            "Choose more salt-tolerant crops or varieties while the soil recovers.",
        ),
        timing="Leach before planting, and keep drains clear through the season.",
        source="FAO Soils Bulletin 39: leaching with adequate drainage",
        cluster="sahel",
    ),
)

_PREFIXES = ("", "Please tell me: ", "Quick question — ", "I want to know, ")


def build_counter_prior(
    n_per_conflict: int = 12, seed: int = 42
) -> list[dict]:
    """Expand each conflict into several phrasings with contrastive answers."""
    rng = random.Random(seed)
    out: list[dict] = []
    for c in CONFLICTS:
        for i in range(n_per_conflict):
            q = c.questions[i % len(c.questions)]
            prefix = _PREFIXES[(i // len(c.questions)) % len(_PREFIXES)]
            # rotate the action order so repeated exposures are not identical
            actions = list(c.actions)
            rng.shuffle(actions)
            answer = render_markdown(
                issue=f"{c.wrong} {c.right}",
                actions=actions,
                timing=c.timing,
                caution=sample_caution(rng, text=c.right, cluster=c.cluster),
                sources=None,
            )
            out.append({
                "id": f"counterprior:{c.name}:{i}",
                "kind": "counter_prior",
                "q": prefix + q,
                "a": answer,
                "meta": {
                    "source_docs": [],
                    "passage_ids": [],
                    "lang": "en",
                    "critique": "pass",
                    "cluster": c.cluster,
                    "conflict": c.name,
                    "corpus_source": c.source,
                },
            })
    return out


def write_counter_prior(out_path: Path, n_per_conflict: int = 12, seed: int = 42) -> dict:
    rows = build_counter_prior(n_per_conflict, seed)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"counter_prior_rows": len(rows), "conflicts": len(CONFLICTS)}
