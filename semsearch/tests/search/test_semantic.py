import numpy as np
import pytest

from semsearch.index.embeddings import EmbeddingIndex
from semsearch.search.semantic import semantic_scores


def test_semantic_scores_picks_best_chunk_per_document(monkeypatch):
    embedding_index = EmbeddingIndex.from_vectors(
        model_name="test-model",
        chunk_doc_ids=["doc-a", "doc-a", "doc-b"],
        vectors=np.asarray(
            [
                [1.0, 0.0],
                [0.24253563, 0.9701425],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        "semsearch.search.semantic.embed_query",
        lambda query, model_name="test-model": np.asarray([0.0, 1.0], dtype=np.float32),
    )

    scores = semantic_scores("example", embedding_index, ["doc-a", "doc-b", "doc-c"])

    assert scores["doc-a"] == pytest.approx(0.9701425, abs=0.02)
    assert scores["doc-b"] == pytest.approx(1.0, abs=0.02)
    assert "doc-c" not in scores
