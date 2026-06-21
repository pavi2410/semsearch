import numpy as np

from .models import EmbeddingCache, db
from .vector_codec import decode_vectors, encode_vectors, is_quantized_embedding


def load_embedding_payload(content_hash: str) -> bytes | None:
    try:
        row = EmbeddingCache.get_by_id(content_hash)
    except EmbeddingCache.DoesNotExist:
        return None
    payload = row.payload
    if not is_quantized_embedding(payload):
        return None
    return payload


def load_embedding(content_hash: str) -> np.ndarray | None:
    payload = load_embedding_payload(content_hash)
    if payload is None:
        return None
    return decode_vectors(payload)


def save_embedding(content_hash: str, vectors: np.ndarray) -> None:
    EmbeddingCache.replace(
        content_hash=content_hash,
        payload=encode_vectors(np.asarray(vectors, dtype=np.float32)),
    ).execute()


def save_embeddings(items: list[tuple[str, np.ndarray]]) -> None:
    if not items:
        return
    with db.atomic():
        for content_hash, vectors in items:
            save_embedding(content_hash, vectors)
