import time

from rank_bm25 import BM25Okapi

from ..index.nlp import preprocess
from ..storage import init_db
from ..storage.models import SyncPage as Page
from .index_store import load_index
from .ranking import apply_ranking, lexical_match_boost


class SearchResult:
    def __init__(
        self, query_time_ms: float, results: list[tuple[str, float]], total_docs: int
    ):
        self.query_time_ms = query_time_ms
        self.results = results
        self.total_docs = total_docs


def _load_search_data() -> tuple[BM25Okapi, list[str], dict[str, float], dict[str, dict[str, str]]]:
    bm25, doc_ids, pagerank = load_index()
    docs = {
        p.url_hash: {
            "url": p.url,
            "title": p.title or "",
            "description": p.description or "",
            "body_excerpt": p.body_excerpt or "",
            "published_at": p.published_at or "",
            "modified_at": p.modified_at or "",
            "fetched_at": p.fetched_at or "",
            "language": p.language or "",
        }
        for p in Page.select(
            Page.url_hash,
            Page.url,
            Page.title,
            Page.description,
            Page.body_excerpt,
            Page.published_at,
            Page.modified_at,
            Page.fetched_at,
            Page.language,
        )
    }
    return bm25, doc_ids, pagerank, docs


_index_loaded = False
_bm25: BM25Okapi | None = None
_doc_ids: list[str] = []
_docs: dict[str, dict[str, str]] = {}
_pagerank: dict[str, float] = {}


def _ensure_index() -> None:
    global _bm25, _doc_ids, _docs, _pagerank, _index_loaded
    if _index_loaded:
        return
    init_db()
    _bm25, _doc_ids, _pagerank, _docs = _load_search_data()
    _index_loaded = True


def get_docs() -> dict[str, dict[str, str]]:
    _ensure_index()
    return _docs


def search(query: str) -> SearchResult:
    _ensure_index()
    start = time.perf_counter_ns()
    query_tokens = preprocess(query)
    scores = _bm25.get_scores(query_tokens)  # type: ignore[union-attr]
    end = time.perf_counter_ns()

    positive_scores = [score for score in scores if score > 0]
    bm25_max = max(positive_scores) if positive_scores else 0.0

    results = []
    for doc_id, bm25_score in zip(_doc_ids, scores):
        if bm25_score <= 0:
            continue
        doc = _docs.get(doc_id, {})
        pagerank_boost = _pagerank.get(doc_id, 1.0)
        results.append(
            (
                doc_id,
                apply_ranking(
                    bm25_score,
                    doc,
                    pagerank_boost=pagerank_boost,
                    bm25_max=bm25_max,
                )
                * lexical_match_boost(query, query_tokens, doc),
            )
        )
    results.sort(key=lambda x: x[1], reverse=True)

    return SearchResult(
        query_time_ms=(end - start) / 1_000_000,
        results=results,
        total_docs=len(_doc_ids),
    )
