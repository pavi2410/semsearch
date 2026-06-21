import numpy as np

from semsearch.storage.vector_codec import decode_vectors, encode_vectors, is_quantized_embedding


def test_encode_decode_round_trip():
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    payload = encode_vectors(vectors)
    restored = decode_vectors(payload)

    assert is_quantized_embedding(payload)
    assert restored.shape == vectors.shape
    assert np.allclose(restored, vectors, atol=0.02)

    large = np.random.default_rng(0).normal(size=(128, 384)).astype(np.float32)
    large /= np.linalg.norm(large, axis=1, keepdims=True)
    large_payload = encode_vectors(large)
    assert len(large_payload) < large.nbytes


def test_encode_empty_vectors():
    payload = encode_vectors(np.empty((0, 384), dtype=np.float32))
    restored = decode_vectors(payload)
    assert restored.shape == (0, 384)


def test_decode_rejects_invalid_payload():
    try:
        decode_vectors(b"not-emb1")
    except ValueError as exc:
        assert "EMB1" in str(exc)
    else:
        raise AssertionError("expected ValueError")
