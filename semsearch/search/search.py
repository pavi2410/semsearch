import time
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from ..index.nlp import preprocess
from ..storage import init_db
from ..storage.models import SyncPage as Page
from .dedup import dedupe_results
from .fusion import reciprocal_rank_fusion
from .index_store import load_index
from .ranking import (
    dampen_metadata_boost,
    https_boost,
    lexical_match_boost,
    metadata_multiplier,
    recency_boost,
)
from .semantic import semantic_scores

SEMANTIC_BM25_SCALE = 8.0


@dataclass(frozen=True)
class ScoreBreakdown:
    bm25: float
    semantic: float
    base: float
    base_source: str
    recency: float
    https: float
    pagerank: float
    metadata: float
    title: float
    fusion: float
    fusion_multiplier: float
    final: float


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float
    breakdown: ScoreBreakdown


class SearchResult:
    def __init__(
        self,
        query_time_ms: float,
        results: list[SearchHit],
        total_docs: int,
    ):
        self.query_time_ms = query_time_ms
        self.results = results
        self.total_docs = total_docs


def format_score_breakdown(breakdown: ScoreBreakdown) -> str:
    parts = [f"score {breakdown.final:.3f}"]

    if breakdown.bm25 > 0:
        parts.append(f"bm25 {breakdown.bm25:.2f}")
    if breakdown.semantic > 0:
        parts.append(f"sem {breakdown.semantic:.3f}")
    if breakdown.semantic > 0 or breakdown.bm25 > 0:
        parts.append(f"base {breakdown.base_source}")

    if abs(breakdown.recency - 1.0) > 0.001:
        parts.append(f"recency×{breakdown.recency:.2f}")
    if abs(breakdown.https - 1.0) > 0.001:
        parts.append(f"https×{breakdown.https:.2f}")
    if abs(breakdown.pagerank - 1.0) > 0.001:
        parts.append(f"pagerank×{breakdown.pagerank:.2f}")
    if abs(breakdown.title - 1.0) > 0.001:
        parts.append(f"title×{breakdown.title:.2f}")
    if breakdown.fusion > 0:
        parts.append(f"fusion×{breakdown.fusion_multiplier:.3f}")

    return " · ".join(parts)


def _score_document(
    doc_id: str,
    *,
    query: str,
    query_tokens: list[str],
    bm25_score: float,
    semantic_score: float,
    bm25_max: float,
    fusion_boost: float,
    pagerank_boost: float,
    doc: dict[str, str],
) -> SearchHit | None:
    semantic_scaled = semantic_score * SEMANTIC_BM25_SCALE
    base_score = max(bm25_score, semantic_scaled)
    if base_score <= 0:
        return None

    base_source = "sem" if semantic_scaled > bm25_score else "bm25"
    recency = recency_boost(doc)
    secure = https_boost(doc)
    meta_raw = metadata_multiplier(doc, pagerank_boost=pagerank_boost)
    relevance = min(1.0, base_score / bm25_max) if bm25_max > 0 else 0.0
    metadata = dampen_metadata_boost(meta_raw, relevance) if bm25_max > 0 else meta_raw
    title = lexical_match_boost(query, query_tokens, doc)
    fusion_multiplier = 1.0 + fusion_boost if fusion_boost else 1.0
    final = base_score * metadata * title * fusion_multiplier

    breakdown = ScoreBreakdown(
        bm25=bm25_score,
        semantic=semantic_score,
        base=base_score,
        base_source=base_source,
        recency=recency,
        https=secure,
        pagerank=pagerank_boost,
        metadata=metadata,
        title=title,
        fusion=fusion_boost,
        fusion_multiplier=fusion_multiplier,
        final=final,
    )
    return SearchHit(doc_id=doc_id, score=final, breakdown=breakdown)


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
    results: list[SearchHit] = []
    for doc_id in candidate_ids:
        doc = _docs.get(doc_id, {})
        hit = _score_document(
            doc_id,
            query=query,
            query_tokens=query_tokens,
            bm25_score=score_by_doc[doc_id],
            semantic_score=semantic.get(doc_id, 0.0),
            bm25_max=bm25_max,
            fusion_boost=fused_scores.get(doc_id, 0.0),
            pagerank_boost=_pagerank.get(doc_id, 1.0),
            doc=doc,
        )
        if hit is not None:
            results.append(hit)

    results.sort(key=lambda hit: hit.score, reverse=True)
    results = dedupe_results(results, _docs)

    return SearchResult(
        query_time_ms=(end - start) / 1_000_000,
        results=results,
        total_docs=len(_doc_ids),
    )
