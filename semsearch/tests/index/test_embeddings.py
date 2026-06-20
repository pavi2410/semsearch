import numpy as np

from semsearch.index.embeddings import embed_text_chunks


class FakeEmbedder:
    last_kwargs: dict | None = None

    def embed(self, chunks, *, batch_size=256, parallel=None):
        FakeEmbedder.last_kwargs = {
            "batch_size": batch_size,
            "parallel": parallel,
        }
        del parallel
        for chunk in chunks:
            value = float(len(chunk))
            yield np.asarray([value, value + 1], dtype=np.float32)


def test_embed_text_chunks_batches_and_normalizes():
    embedder = FakeEmbedder()
    vectors = embed_text_chunks(["aa", "bbbb"], embedder)

    assert vectors.shape == (2, 2)
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)
    assert np.isclose(np.linalg.norm(vectors[1]), 1.0)
    assert embedder.last_kwargs == {"batch_size": 64, "parallel": None}


def test_embed_text_chunks_can_opt_in_to_parallel():
    embedder = FakeEmbedder()
    embed_text_chunks(["aa"], embedder, parallel=4)

    assert embedder.last_kwargs == {"batch_size": 64, "parallel": 4}
