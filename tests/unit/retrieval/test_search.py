from collections import Counter

from sahel_sage.retrieval.evidence import build_pack
from sahel_sage.retrieval.store import NullRetriever, Retriever, open_retriever


def test_accent_insensitive_french_match(library):
    # unaccented farmer spelling must find the accented French manual
    db_path, _ = library
    r = Retriever(db_path)
    hits = r.search("diarrhee chevre")
    assert hits and hits[0].doc_id == "goat_fr"
    assert "diarrhée" in hits[0].text


def test_diversity_cap_max_two_chunks_per_doc(library):
    db_path, _ = library
    r = Retriever(db_path)
    hits = r.search("millet downy mildew disease", k=4)
    assert hits
    per_doc = Counter(h.doc_id for h in hits)
    assert max(per_doc.values()) <= 2


def test_evidence_pack_empty_is_insufficient():
    pack = build_pack([], threshold=0.35, k=4)
    assert pack.confidence == 0.0
    assert pack.sufficient is False
    assert pack.items == []


def test_evidence_pack_strong_query_is_sufficient(library):
    db_path, _ = library
    r = Retriever(db_path)
    hits = r.search("millet downy mildew", k=4)
    pack = build_pack(hits, threshold=0.35, k=4)
    assert pack.sufficient is True
    assert 0.0 < pack.confidence <= 1.0
    assert pack.items == hits


def test_stats_and_documents(library):
    db_path, stats = library
    r = Retriever(db_path)
    s = r.stats()
    assert s["documents"] == stats["docs"]
    assert s["chunks"] == stats["chunks"]
    assert "chunker" in s
    docs = r.list_documents()
    assert {d["id"] for d in docs} == {"millet_mildew", "goat_fr", "maize_storage"}
    assert all(d["chunks"] >= 1 for d in docs)


def test_open_retriever_fallback(tmp_path, library):
    assert isinstance(open_retriever(tmp_path / "missing.db"), NullRetriever)
    assert isinstance(open_retriever(library[0]), Retriever)
    null = NullRetriever()
    assert null.search("anything") == []
    assert null.stats()["documents"] == 0
    assert null.list_documents() == []
