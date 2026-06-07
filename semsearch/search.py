import json
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from .nlp import preprocess

DOCS_FILE = Path("docs.json")
INDEX_FILE = Path("index_state.json")


class SearchResult:
    def __init__(self, query_time: float, results: list[tuple[str, float]]):
        self.query_time = query_time
        self.results = results


def _load_index() -> tuple[BM25Okapi, list[str], dict[str, dict[str, str]]]:
    with open(INDEX_FILE, encoding="utf-8") as f:
        params = json.load(f)

    doc_ids: list[str] = params["doc_ids"]
    corpus_tokens: list[list[str]] = params["corpus_tokens"]
    bm25 = BM25Okapi(corpus_tokens, k1=params["k1"], b=params["b"])

    with open(DOCS_FILE, encoding="utf-8") as f:
        docs: dict[str, dict[str, str]] = json.load(f)

    return bm25, doc_ids, docs


_index_loaded = False
_bm25: BM25Okapi | None = None
_doc_ids: list[str] = []
_docs: dict[str, dict[str, str]] = {}


def _ensure_index() -> None:
    global _bm25, _doc_ids, _docs, _index_loaded
    if _index_loaded:
        return
    _bm25, _doc_ids, _docs = _load_index()
    _index_loaded = True


def get_docs() -> dict[str, dict[str, str]]:
    _ensure_index()
    return _docs


def search(query: str) -> SearchResult:
    _ensure_index()
    start = time.perf_counter()
    query_tokens = preprocess(query)
    scores = _bm25.get_scores(query_tokens)  # type: ignore[union-attr]
    end = time.perf_counter()

    paired = list(zip(_doc_ids, scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    filtered = [(did, s) for did, s in paired if s > 0]

    return SearchResult(
        query_time=(end - start) * 1000,
        results=filtered,
    )
