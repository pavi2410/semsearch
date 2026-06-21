import numpy as np

from .models import EmbeddingCache
from .vector_codec import decode_vectors, encode_vectors, is_quantized_embedding


def load_embedding(content_hash: str) -> np.ndarray | None:
    try:
        row = EmbeddingCache.get_by_id(content_hash)
    except EmbeddingCache.DoesNotExist:
        return None
    payload = row.payload
    if not is_quantized_embedding(payload):
        return None
    return decode_vectors(payload)


def save_embedding(content_hash: str, vectors: np.ndarray) -> None:
    EmbeddingCache.replace(
        content_hash=content_hash,
        payload=encode_vectors(np.asarray(vectors, dtype=np.float32)),
    ).execute()
