"""FAO terminology must survive translation, and the near-misses must not.

Translation quality is not what loses the African-language bonus; terminology
is. A sentence can be fluent, grammatical, natural Arabic and still name a
different plant, and a reviewer checking that the Arabic reads well will pass
it. An agronomist judge will not.
"""

from __future__ import annotations

import pytest

from sahel_sage.training.glossary import check, load_glossary, normalize_ar

#: The pair that motivated the whole module. Both are real Arabic terms for real
#: parasitic weeds of cereals; only one of them is Striga.
STRIGA_AR = "ستريجا"
OROBANCHE_AR = ("الهالوك", "جعفيل")


def test_glossary_has_the_terms_the_twelve_topics_need() -> None:
    g = load_glossary()
    for term in ("Striga", "Newcastle disease", "peste des petits ruminants",
                 "pearl millet", "sorghum", "cowpeas", "aflatoxins"):
        assert term in g and g[term]["ar"], f"{term} missing from the glossary"


@pytest.mark.parametrize("wrong_ar", OROBANCHE_AR)
def test_orobanche_is_rejected_for_striga(wrong_ar: str) -> None:
    """The substitution a fluency reviewer would never catch."""
    c = check("Pull the Striga out by hand.", f"انزع {wrong_ar} باليد.")
    assert not c.ok
    assert any(w[0] == wrong_ar for w in c.wrong)


def test_correct_striga_term_passes() -> None:
    assert check("Pull the Striga out by hand.", f"انزع ال{STRIGA_AR} باليد.").ok


def test_the_two_diseases_are_not_interchangeable() -> None:
    """Newcastle is a poultry virus; PPR is a small-ruminant one.

    Our English model confused exactly these before the reference block existed,
    so the Arabic path must not be free to repeat it.
    """
    ppr_ar = "طاعون المجترات الصغيرة"
    newcastle_ar = "مرض نيوكاسل"
    assert not check("Vaccinate against Newcastle disease.", f"لقّح ضد {ppr_ar}.").ok
    assert not check("This is peste des petits ruminants.", f"هذا {newcastle_ar}.").ok
    assert check("Vaccinate against Newcastle disease.", f"لقّح ضد {newcastle_ar}.").ok


def test_a_missing_term_is_caught() -> None:
    """Vague-but-fluent is still a failure: "grain" is not "pearl millet"."""
    c = check("Sow pearl millet at the onset of the rains.",
              "ازرع الحبوب مع بداية الأمطار.")
    assert not c.ok
    assert "pearl millet" in c.missing


class TestArabicNormalisation:
    """Arabic fuses its article and prepositions onto the stem.

    Without folding them the check rejects correct translations wholesale:
    AGROVOC gives `فول سوداني`, a real sentence says `الفول السوداني`, and the
    bare form is not a substring of the definite one because `ال` attaches to
    *both* words.
    """

    def test_definite_article_does_not_defeat_a_match(self) -> None:
        assert check("Dry the groundnuts.", "جفف الفول السوداني.").ok

    def test_stacked_clitics_are_stripped(self) -> None:
        assert normalize_ar("وبالأفلاتوكسين").endswith("افلاتوكسين")

    def test_orthographic_variants_fold(self) -> None:
        # alif forms and ta-marbuta are written inconsistently by any translator
        assert normalize_ar("أفلاتوكسين") == normalize_ar("افلاتوكسين")

    def test_tashkil_does_not_defeat_a_match(self) -> None:
        # we never emit short vowels (5.6x the tokens) but input may carry them
        assert check("Vaccinate against Newcastle disease.",
                     "لقّح ضد مَرَض نيوكاسل.").ok


def test_untouched_english_is_not_flagged() -> None:
    """A term absent from the English is not required in the Arabic."""
    assert check("Water the animals regularly.", "اسق الحيوانات بانتظام.").ok
