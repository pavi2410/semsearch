from semsearch.index.embedding_batch import take_embed_batch


def _doc(name: str, chunk_count: int) -> tuple[str, str, list[str]]:
    return (name, f"hash-{name}", [f"{name}-{index}" for index in range(chunk_count)])


def test_take_embed_batch_splits_on_chunk_budget():
    ready = [_doc("a", 20), _doc("b", 20), _doc("c", 20), _doc("d", 5)]

    first = take_embed_batch(ready, chunk_budget=96, solo_doc_chunks=32)

    assert first == [_doc("a", 20), _doc("b", 20), _doc("c", 20)]
    assert ready == [_doc("d", 5)]


def test_take_embed_batch_embeds_large_docs_solo():
    ready = [_doc("big", 50), _doc("small", 5)]

    first = take_embed_batch(ready, chunk_budget=96, solo_doc_chunks=32)

    assert first == [_doc("big", 50)]
    assert ready == [_doc("small", 5)]
