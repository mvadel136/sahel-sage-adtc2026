from sahel_sage.data.kallaama import clean_segment


def test_fra_markers_dropped_words_kept():
    assert clean_segment("jiwu semence :fra certifiée :fra bu baax") == "jiwu semence certifiée bu baax"


def test_event_annotations_removed():
    assert clean_segment("waaw [rire] ñu ngi koy wax (musique)") == "waaw ñu ngi koy wax"
