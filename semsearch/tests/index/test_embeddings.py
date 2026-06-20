import numpy as np

from semsearch.index.embeddings import embed_text_chunks


class FakeEmbedder:
    def embed(self, chunks, *, batch_size=256, parallel=None):
        del batch_size, parallel
        for chunk in chunks:
            value = float(len(chunk))
            yield np.asarray([value, value + 1], dtype=np.float32)


def test_embed_text_chunks_batches_and_normalizes():
    vectors = embed_text_chunks(["aa", "bbbb"], FakeEmbedder())

    assert vectors.shape == (2, 2)
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)
    assert np.isclose(np.linalg.norm(vectors[1]), 1.0)
