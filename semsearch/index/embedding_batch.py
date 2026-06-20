def take_embed_batch(
    ready: list[tuple[str, str, list[str]]],
    *,
    chunk_budget: int,
    solo_doc_chunks: int,
) -> list[tuple[str, str, list[str]]]:
    """Take the next ONNX batch, bounding total chunks to avoid long stalls."""
    if not ready:
        return []

    if len(ready[0][2]) >= solo_doc_chunks:
        return [ready.pop(0)]

    batch: list[tuple[str, str, list[str]]] = []
    chunk_count = 0

    while ready:
        doc = ready[0]
        chunk_count_for_doc = len(doc[2])
        if chunk_count_for_doc >= solo_doc_chunks:
            break
        if batch and chunk_count + chunk_count_for_doc > chunk_budget:
            break
        batch.append(ready.pop(0))
        chunk_count += chunk_count_for_doc

    return batch
