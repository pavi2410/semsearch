import numpy as np

from ..index.embeddings import EmbeddingIndex, embed_query


def semantic_scores(
    query: str,
    embedding_index: EmbeddingIndex,
    doc_ids: list[str],
) -> dict[str, float]:
    """Return the best chunk similarity per document."""
    if embedding_index.vectors.size == 0:
        return {}

    query_vector = embed_query(query, model_name=embedding_index.model_name)
    chunk_scores = embedding_index.vectors @ query_vector

    best_scores = {doc_id: 0.0 for doc_id in doc_ids}
    for doc_id, score in zip(embedding_index.chunk_doc_ids, chunk_scores, strict=True):
        if score > best_scores.get(doc_id, 0.0):
            best_scores[doc_id] = float(score)

    return {doc_id: score for doc_id, score in best_scores.items() if score > 0}
