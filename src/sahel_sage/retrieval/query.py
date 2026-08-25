"""Query-side text handling: tokenizing, stopwords, synonyms, FTS5 sanitizing.

Recall matters more than elegance here. A farmer writes "my goats have
diarrhea", the manual says "enteritis in small ruminants", pure lexical search
misses that, so queries are expanded with a small curated agronomy synonym map
before they hit FTS5. The map is deliberately small: a large auto-generated map
adds noise faster than recall. French equivalents sit beside the English
entries because the corpus and the farmers are bilingual; ``remove_diacritics
2`` in the index makes accented and unaccented forms equivalent.

Every term is re-emitted double-quoted before reaching FTS5, so raw operator
characters typed by a user (``- " * : ( )``, ``AND``, ``NEAR(`` ...) can never
cause an fts5 syntax error.
"""

from __future__ import annotations

import re

# Farmer vocabulary -> manual vocabulary, English + French.
SYNONYMS: dict[str, list[str]] = {
    "diarrhea": ["diarrhoea", "scouring", "enteritis", "loose faeces", "diarrhée"],
    "diarrhoea": ["diarrhea", "scouring", "enteritis", "diarrhée"],
    "diarrhée": ["diarrhea", "scouring", "enteritis", "entérite"],
    "worms": ["helminth", "parasite", "nematode", "deworming", "anthelmintic", "vers", "vermifuge"],
    "vers": ["helminth", "parasite", "worms", "vermifuge"],
    "cough": ["respiratory", "pneumonia", "coughing", "toux"],
    "toux": ["respiratoire", "pneumonie", "cough"],
    "bugs": ["insect", "pest", "infestation", "insecte", "ravageur"],
    "insects": ["pest", "infestation", "larvae", "insecte", "ravageur"],
    "insectes": ["pest", "infestation", "larves", "ravageur"],
    "caterpillar": ["armyworm", "larvae", "borer", "chenille"],
    "chenille": ["armyworm", "larvae", "caterpillar", "légionnaire"],
    "armyworm": ["spodoptera", "fall armyworm", "larvae", "chenille légionnaire"],
    "weevil": ["storage pest", "sitophilus", "bruchid", "charançon"],
    "charançon": ["weevil", "sitophilus", "bruchid"],
    "mold": ["mould", "fungal", "aflatoxin", "mycotoxin", "moisissure"],
    "mould": ["fungal", "aflatoxin", "mycotoxin", "moisissure"],
    "moisissure": ["mould", "fungal", "aflatoxine", "mycotoxine"],
    "yellowing": ["chlorosis", "nutrient deficiency", "yellow leaves", "jaunissement"],
    "jaunissement": ["chlorose", "carence", "yellowing"],
    "wilting": ["wilt", "water stress", "vascular", "flétrissement"],
    "fertilizer": ["fertiliser", "npk", "urea", "nutrient application", "engrais"],
    "engrais": ["fertilizer", "npk", "urée", "fertilisation"],
    "manure": ["organic matter", "compost", "farmyard manure", "fumier"],
    "fumier": ["manure", "compost", "matière organique"],
    "goat": ["goats", "small ruminant", "caprine", "chèvre"],
    "goats": ["small ruminant", "caprine", "chèvre"],
    "chèvre": ["goat", "small ruminant", "caprin"],
    "chèvres": ["goat", "small ruminant", "caprin"],
    "sheep": ["small ruminant", "ovine", "ewe", "lamb", "mouton", "brebis"],
    "mouton": ["sheep", "small ruminant", "ovin", "brebis"],
    "cow": ["cattle", "bovine", "heifer", "vache"],
    "cows": ["cattle", "bovine", "vache"],
    "vache": ["cattle", "bovine", "cow", "bovin"],
    "cattle": ["bovine", "cow", "bovin", "vache"],
    "camel": ["camels", "dromedary", "camelid", "chameau", "dromadaire"],
    "chameau": ["camel", "dromedary", "dromadaire"],
    "chicken": ["poultry", "chickens", "hens", "village poultry", "poule", "poulet"],
    "poule": ["chicken", "poultry", "volaille", "poulet"],
    "poulet": ["chicken", "poultry", "volaille", "poule"],
    "hens": ["poultry", "layers", "laying hens", "poule pondeuse"],
    "millet": ["pearl millet", "pennisetum", "mil"],
    "mil": ["millet", "pearl millet", "pennisetum"],
    "corn": ["maize", "maïs"],
    "maize": ["corn", "maïs"],
    "maïs": ["maize", "corn"],
    "storage": ["store", "granary", "hermetic", "post-harvest", "stockage", "grenier"],
    "stockage": ["storage", "granary", "hermetic", "grenier", "conservation"],
    "drought": ["dry spell", "water stress", "rainfall deficit", "sécheresse"],
    "sécheresse": ["drought", "dry spell", "stress hydrique"],
    "rain": ["rainfall", "rainy season", "precipitation", "pluie"],
    "pluie": ["rainfall", "rainy season", "précipitations", "hivernage"],
    "soil": ["soil fertility", "land", "topsoil", "sol"],
    "sol": ["soil", "fertilité", "terre"],
    "salt": ["salinity", "saline", "sodicity", "salinité"],
    "price": ["market", "marketing", "income", "prix", "marché"],
    "prix": ["market", "marketing", "price", "marché"],
    "seed": ["seeds", "sowing", "planting", "semence", "semis"],
    "seeds": ["sowing", "planting", "semence", "semis"],
    "semence": ["seed", "sowing", "semis", "variété"],
    "semis": ["sowing", "planting", "semence"],
}

STOP = {
    "the", "and", "for", "with", "what", "how", "why", "should", "would",
    "have", "has", "are", "is", "was", "were", "this", "that", "them",
    "they", "there", "their", "from", "some", "any", "can", "could", "will",
    "about", "into", "than", "then", "when", "which", "who", "does", "did",
    "doing", "done", "very", "much", "many", "more", "most", "also", "but",
    "not", "you", "your", "my", "our", "his", "her", "its", "it", "in",
    "on", "of", "to", "at", "by", "as", "be", "been", "being", "do",
}  # fmt: skip

_WORD = re.compile(r"[a-zà-ÿ0-9']+")


def tokenize(q: str) -> list[str]:
    """Lowercased word tokens, minus stopwords and tokens of <= 2 chars."""
    return [w for w in _WORD.findall(q.lower()) if w not in STOP and len(w) > 2]


def expand(terms: list[str]) -> list[str]:
    """Terms plus their synonyms, order-preserving and de-duplicated."""
    out: list[str] = []
    for t in terms:
        out.append(t)
        out.extend(SYNONYMS.get(t, []))
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def quote(term: str) -> str:
    """Double-quote a term (or multi-word synonym) as an FTS5 string/phrase.

    Tokenized terms cannot contain a double quote, but synonyms are free text,
    so embedded quotes are doubled per SQL string rules.
    """
    return '"' + term.replace('"', '""') + '"'


def sanitize_fts_query(user_text: str) -> str:
    """User text -> FTS5 MATCH string that can never raise a syntax error.

    Tokenizes with the same regex as retrieval, drops stopwords and short
    tokens, and re-emits each term double-quoted (implicit AND between terms).
    May return "" when nothing survives, callers must treat that as no query.
    """
    return " ".join(quote(t) for t in tokenize(user_text))


def match_variants(question: str) -> list[str]:
    """The three lexical retrieval legs, built from sanitized quoted terms.

    1. all terms OR'd (recall), 2. synonym-expanded OR'd (vocabulary gap),
    3. the two longest terms AND'd (precision leg for specific entities).
    """
    terms = tokenize(question)
    if not terms:
        return []
    variants = [
        " OR ".join(quote(t) for t in terms),
        " OR ".join(quote(t) for t in expand(terms)),
    ]
    if len(terms) >= 2:
        rare = sorted(terms, key=len, reverse=True)[:2]
        variants.append(" AND ".join(quote(t) for t in rare))
    return variants
