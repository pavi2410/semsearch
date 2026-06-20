from semsearch.index.chunking import chunk_text


def test_chunk_text_splits_long_paragraphs():
    text = "First paragraph.\n\n" + ("word " * 300)
    chunks = chunk_text(text, max_chars=200)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_chunk_text_merges_short_paragraphs():
    chunks = chunk_text("Alpha paragraph.\n\nBeta paragraph.", max_chars=200)

    assert chunks == ["Alpha paragraph.\nBeta paragraph."]
