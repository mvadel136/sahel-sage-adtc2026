"""Shared fixture: a tiny 3-doc corpus indexed into a tmp SQLite library.

One English doc long enough to produce several chunks (millet / downy mildew),
one French doc with accented words (goat diarrhoea), one English maize-storage
doc, plus a too-short file that the indexer must skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sahel_sage.retrieval.indexer import build_index

_MILLET_PARA = (
    "Pearl millet is the staple cereal of the Sahel and downy mildew is its most "
    "damaging disease in the rainy season. Infected millet seedlings show pale "
    "chlorotic leaves and twisted green ears. Farmers should uproot and burn "
    "infected millet plants early, choose resistant pearl millet varieties, and "
    "treat seed before sowing. Rotating millet with cowpea reduces the downy "
    "mildew inoculum that survives in the soil between seasons. Field number {i} "
    "in the demonstration trial confirmed these downy mildew control practices."
)

_GOAT_FR = """La diarrhée chez la chèvre

La diarrhée est un signe fréquent de maladie chez la chèvre au Sahel. Une chèvre
atteinte de diarrhée perd rapidement de l'eau et des sels minéraux, surtout chez
les chevreaux. Donnez une solution de réhydratation et gardez la chèvre à l'ombre.

Si la diarrhée persiste plus de deux jours, la cause peut être une entérite ou
des parasites internes. Un vermifuge adapté aux petits ruminants aide souvent.
Consultez l'agent vétérinaire du village avant tout traitement antibiotique.

Pour prévenir la diarrhée, gardez l'enclos des chèvres propre et sec, donnez de
l'eau propre chaque jour et évitez les changements brusques d'alimentation qui
perturbent le rumen des petits ruminants et provoquent des troubles digestifs.
"""

_MAIZE = """Storing maize after harvest

Maize grain must be dried to below fourteen percent moisture before storage or
mould and weevils will destroy it within weeks. Dry the cobs on raised racks in
the sun and test dryness by biting a kernel: a dry kernel cracks cleanly.

Hermetic storage bags keep insects out without chemicals. Fill the bag
completely with dry maize grain, press out the air, and tie each liner
separately. A well sealed hermetic bag protects maize for a full year in the
granary, preserving both food and seed for the next planting season.
"""


@pytest.fixture(scope="session")
def library(tmp_path_factory) -> tuple[Path, dict]:
    root = tmp_path_factory.mktemp("corpus")
    txt = root / "txt"
    txt.mkdir()
    millet = "Millet downy mildew control\n\n" + "\n\n".join(
        _MILLET_PARA.format(i=i) for i in range(12)
    )
    (txt / "millet_mildew.txt").write_text(millet)
    (txt / "goat_fr.txt").write_text(_GOAT_FR)
    (txt / "maize_storage.txt").write_text(_MAIZE)
    (txt / "tiny.txt").write_text("Too short to index.")
    sources = [
        {"id": "millet_mildew", "title": "Millet Diseases", "org": "ICRISAT", "lang": "en"},
        {"id": "goat_fr", "title": "Santé des chèvres", "org": "FAO", "lang": "fr"},
        {"id": "maize_storage", "title": "Post-harvest Handling", "org": "FAO", "lang": "en"},
    ]
    sources_json = root / "corpus_sources.json"
    sources_json.write_text(json.dumps(sources))
    db_path = root / "library.db"
    stats = build_index(db_path, txt, sources_json)
    return db_path, stats
