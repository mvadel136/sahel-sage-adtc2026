from sahel_sage.core import textproc

GARBAGE = "WKHSURGXFWLYLWDQGSURWDELOLWRIWKHIDUP"
REAL = "Sorghum is a drought-tolerant cereal grown across the Sahel."


def test_garbage_line_rejected():
    assert not textproc.line_is_text(GARBAGE)
    assert textproc.line_is_text(REAL)


def test_clean_drops_repeated_headers():
    pages = [
        f"Manual Title Header\nreal content number {i} about millet farming practices\n"
        for i in range(6)
    ]
    cleaned = textproc.clean_extracted_text("".join(pages))
    assert "Manual Title Header" not in cleaned
    assert "millet" in cleaned


def test_sections_split_on_headings():
    text = "Intro text before headings.\n\nControlling Storage Pests\nUse airtight bags.\n"
    secs = textproc.split_sections(text)
    assert len(secs) == 2
    assert secs[1][0] == "Controlling Storage Pests"


def test_chunk_overlap_and_min_words():
    para = " ".join(["word"] * 100)
    body = "\n\n".join([para] * 5)
    chunks = textproc.chunk(body, target_words=220, overlap_words=40, min_words=25)
    assert all(len(c.split()) >= 25 for c in chunks)
    assert len(chunks) >= 2
    # tail of chunk 1 overlaps head of chunk 2
    tail = " ".join(chunks[0].split()[-40:])
    assert chunks[1].startswith(tail)


def test_giant_paragraph_is_hard_split():
    # one 5000-word "paragraph" (no blank lines) must not become one chunk
    body = " ".join(f"w{i}" for i in range(5000))
    chunks = textproc.chunk(body, target_words=700, overlap_words=80, min_words=120)
    assert len(chunks) >= 6
    # soft bound: a chunk may carry the overlap tail on top of one full window
    assert all(len(c.split()) <= 700 + 80 for c in chunks)


def test_chunk_id_stable():
    a = textproc.chunk_id("doc", 0, "hello world")
    b = textproc.chunk_id("doc", 0, "hello world")
    assert a == b and a.startswith("doc:0:")


def test_iter_doc_chunks_excludes_holdout(tmp_path):
    (tmp_path / "keep.txt").write_text(" ".join(["alpha"] * 200))
    (tmp_path / "held.txt").write_text(" ".join(["beta"] * 200))
    ids = [
        cid
        for cid, _ in textproc.iter_doc_chunks(
            tmp_path, target_words=100, overlap_words=0, min_words=10, exclude_docs={"held"}
        )
    ]
    assert ids and all(cid.startswith("keep:") for cid in ids)
