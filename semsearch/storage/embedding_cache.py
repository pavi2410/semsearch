import pickle

import numpy as np

from .models import SyncEmbeddingCache as EmbeddingCache


def load_embedding(content_hash: str) -> np.ndarray | None:
    try:
        row = EmbeddingCache.get_by_id(content_hash)
    except EmbeddingCache.DoesNotExist:
        return None
    payload = pickle.loads(row.payload)
    if isinstance(payload, dict):
        return None
    return np.asarray(payload, dtype=np.float32)


def save_embedding(content_hash: str, vectors: np.ndarray) -> None:
    EmbeddingCache.replace(
        content_hash=content_hash,
        payload=pickle.dumps(np.asarray(vectors, dtype=np.float32), protocol=pickle.HIGHEST_PROTOCOL),
    ).execute()
