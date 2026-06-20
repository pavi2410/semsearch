import time

from rank_bm25 import BM25Okapi

from ..index.nlp import preprocess
from ..storage import init_db
from ..storage.models import SyncPage as Page
from .dedup import dedupe_results
from .fusion import reciprocal_rank_fusion
from .index_store import load_index
from .ranking import apply_ranking, lexical_match_boost
from .semantic import semantic_scores

SEMANTIC_BM25_SCALE = 8.0


class SearchResult:
    def __init__(
        self, query_time_ms: float, results: list[tuple[str, float]], total_docs: int
    ):
        self.query_time_ms = query_time_ms
        self.results = results
        self.total_docs = total_docs


def _load_search_data() -> tuple[
    BM25Okapi,
    list[str],
    dict[str, float],
    object | None,
    dict[str, dict[str, str]],
]:
    bm25, doc_ids, pagerank, embedding_index = load_index()
    docs = {
        p.url_hash: {
            "url": p.url,
            "canonical_url": p.canonical_url or "",
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
            Page.canonical_url,
            Page.title,
            Page.description,
            Page.body_excerpt,
            Page.published_at,
            Page.modified_at,
            Page.fetched_at,
            Page.language,
        )
    }
    return bm25, doc_ids, pagerank, embedding_index, docs


_index_loaded = False
_bm25: BM25Okapi | None = None
_doc_ids: list[str] = []
_docs: dict[str, dict[str, str]] = {}
_pagerank: dict[str, float] = {}
_embedding_index = None


def _ensure_index() -> None:
    global _bm25, _doc_ids, _docs, _pagerank, _embedding_index, _index_loaded
    if _index_loaded:
        return
    init_db()
    _bm25, _doc_ids, _pagerank, _embedding_index, _docs = _load_search_data()
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

    semantic = (
        semantic_scores(query, _embedding_index, _doc_ids)
        if _embedding_index is not None
        else {}
    )

    score_by_doc = dict(zip(_doc_ids, scores, strict=True))
    bm25_ranking = [
        doc_id for doc_id, score in sorted(score_by_doc.items(), key=lambda item: item[1], reverse=True)
        if score > 0
    ]
    semantic_ranking = sorted(semantic.keys(), key=lambda doc_id: semantic[doc_id], reverse=True)
    fused_scores = reciprocal_rank_fusion(bm25_ranking, semantic_ranking)

    candidate_ids = (
        sorted(fused_scores, key=fused_scores.get, reverse=True)
        if fused_scores
        else bm25_ranking
    )
    results = []
    for doc_id in candidate_ids:
        bm25_score = score_by_doc[doc_id]
        semantic_score = semantic.get(doc_id, 0.0)
        base_score = max(bm25_score, semantic_score * SEMANTIC_BM25_SCALE)
        if base_score <= 0:
            continue

        doc = _docs.get(doc_id, {})
        pagerank_boost = _pagerank.get(doc_id, 1.0)
        score = (
            apply_ranking(
                base_score,
                doc,
                pagerank_boost=pagerank_boost,
                bm25_max=bm25_max,
            )
            * lexical_match_boost(query, query_tokens, doc)
        )
        fusion_boost = fused_scores.get(doc_id, 0.0)
        if fusion_boost:
            score *= 1.0 + fusion_boost
        results.append((doc_id, score))

    results.sort(key=lambda x: x[1], reverse=True)
    results = dedupe_results(results, _docs)

    return SearchResult(
        query_time_ms=(end - start) / 1_000_000,
        results=results,
        total_docs=len(_doc_ids),
    )
