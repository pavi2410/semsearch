from dataclasses import dataclass

import numpy as np
from fastembed import TextEmbedding

from ..crawl.content_filter import is_indexable_page
from ..crawl.metadata import PageMetadata, extract_page_metadata
from ..storage.content import try_read_content
from .chunking import chunk_text
from .embedding_config import EMBED_CHUNK_BATCH_SIZE, EMBED_PARALLEL
from .embedding_model import DEFAULT_MODEL, load_embedder

DEFAULT_EMBED_BATCH_SIZE = EMBED_CHUNK_BATCH_SIZE


@dataclass(frozen=True)
class DocumentEmbedding:
    chunks: list[str]
    vectors: np.ndarray


@dataclass(frozen=True)
class EmbeddingIndex:
    model_name: str
    chunk_doc_ids: list[str]
    vectors: np.ndarray


def build_document_text(page_meta: PageMetadata) -> str:
    return "\n\n".join(
        part
        for part in (page_meta.title, page_meta.description, page_meta.body_text)
        if part
    )


def chunks_for_content(content_hash: str, url: str) -> list[str] | None:
    html = try_read_content(content_hash)
    if html is None:
        return None
    if not is_indexable_page(url, html):
        return None
    page_meta = extract_page_metadata(html, url)
    return chunk_text(build_document_text(page_meta))


def embed_text_chunks(
    chunks: list[str],
    embedder: TextEmbedding,
    *,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    parallel: int | None = None,
) -> np.ndarray:
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    workers = EMBED_PARALLEL if parallel is None else parallel
    embed_kwargs: dict = {
        "batch_size": batch_size,
    }
    if workers is not None:
        embed_kwargs["parallel"] = workers

    vectors = np.asarray(
        list(embedder.embed(chunks, **embed_kwargs)),
        dtype=np.float32,
    )
    return _normalize_rows(vectors)


def embed_document(
    page_meta: PageMetadata,
    *,
    model: TextEmbedding | None = None,
) -> DocumentEmbedding | None:
    chunks = chunk_text(build_document_text(page_meta))
    if not chunks:
        return None

    embedder = model or load_embedder()
    vectors = embed_text_chunks(chunks, embedder)
    return DocumentEmbedding(chunks=chunks, vectors=vectors)


def build_embedding_index(
    doc_ids: list[str],
    doc_embeddings: dict[str, DocumentEmbedding],
    *,
    model_name: str = DEFAULT_MODEL,
) -> EmbeddingIndex | None:
    chunk_doc_ids: list[str] = []
    vector_rows: list[np.ndarray] = []

    for doc_id in doc_ids:
        embedding = doc_embeddings.get(doc_id)
        if embedding is None or embedding.vectors.size == 0:
            continue
        chunk_doc_ids.extend([doc_id] * len(embedding.chunks))
        vector_rows.append(embedding.vectors)

    if not vector_rows:
        return None

    return EmbeddingIndex(
        model_name=model_name,
        chunk_doc_ids=chunk_doc_ids,
        vectors=np.vstack(vector_rows),
    )


def embed_query(query: str, *, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    embedder = load_embedder(model_name=model_name)
    vector = np.asarray(next(embedder.embed([query])), dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms
