import pickle

import numpy as np

from .models import SyncEmbeddingCache as EmbeddingCache


def load_embedding(content_hash: str) -> tuple[list[str], np.ndarray] | None:
    try:
        row = EmbeddingCache.get_by_id(content_hash)
    except EmbeddingCache.DoesNotExist:
        return None
    payload = pickle.loads(row.payload)
    return payload["chunks"], payload["vectors"]


def save_embedding(content_hash: str, chunks: list[str], vectors: np.ndarray) -> None:
    payload = pickle.dumps({"chunks": chunks, "vectors": vectors})
    EmbeddingCache.replace(content_hash=content_hash, payload=payload).execute()
