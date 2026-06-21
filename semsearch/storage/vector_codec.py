import struct

import numpy as np

EMB1_MAGIC = b"EMB1"
EMB1_VERSION = 1
_EMB1_HEADER = struct.Struct("<4sB3xII")


def is_quantized_embedding(payload: bytes) -> bool:
    return len(payload) >= _EMB1_HEADER.size and payload[:4] == EMB1_MAGIC


def encode_vectors(vectors: np.ndarray) -> bytes:
    """Quantize L2-normalized float32 vectors to EMB1 int8 payload."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    nrows, ndim = vectors.shape
    if nrows == 0:
        return _EMB1_HEADER.pack(EMB1_MAGIC, EMB1_VERSION, 0, ndim)
    scales = np.max(np.abs(vectors), axis=1).astype(np.float32)
    scales = np.where(scales == 0, 1.0, scales)
    quantized = np.round(vectors / scales[:, None] * 127.0).astype(np.int32)
    quantized = np.clip(quantized, -127, 127).astype(np.int8)

    header = _EMB1_HEADER.pack(EMB1_MAGIC, EMB1_VERSION, nrows, ndim)
    return header + scales.tobytes() + quantized.tobytes()


def decode_vectors(payload: bytes) -> np.ndarray:
    """Decode an EMB1 payload back to float32 vectors."""
    if not is_quantized_embedding(payload):
        raise ValueError("payload is not a quantized EMB1 embedding")

    _magic, version, nrows, ndim = _EMB1_HEADER.unpack_from(payload)
    if version != EMB1_VERSION:
        raise ValueError(f"unsupported EMB1 version: {version}")

    if nrows == 0:
        return np.empty((0, ndim), dtype=np.float32)

    scales_offset = _EMB1_HEADER.size
    data_offset = scales_offset + (4 * nrows)
    expected_size = data_offset + (nrows * ndim)
    if len(payload) != expected_size:
        raise ValueError("truncated EMB1 payload")

    scales = np.frombuffer(payload, dtype=np.float32, count=nrows, offset=scales_offset)
    quantized = np.frombuffer(payload, dtype=np.int8, count=nrows * ndim, offset=data_offset)
    vectors = quantized.reshape(nrows, ndim).astype(np.float32) * (scales[:, None] / 127.0)
    return _normalize_rows(vectors)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms
