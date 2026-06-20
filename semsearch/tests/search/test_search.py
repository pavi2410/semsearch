from semsearch.search.search import (
    ScoreBreakdown,
    SearchHit,
    _relevance_base,
    _score_document,
    _sort_hits,
    build_rank_maps,
    format_score_breakdown,
    format_score_breakdown_rich,
)


def _sample_breakdown(**overrides) -> ScoreBreakdown:
    defaults = {
        "bm25": 8.0,
        "semantic": 0.712,
        "bm25_rank": 2,
        "semantic_rank": 5,
        "fused_rank": 1,
        "base": 0.032,
        "base_source": "rrf",
        "recency": 1.12,
        "https": 1.05,
        "pagerank": 1.15,
        "metadata": 1.08,
        "title": 1.25,
        "fusion": 0.032,
        "fusion_multiplier": 1.0,
        "final": 0.043,
    }
    defaults.update(overrides)
    return ScoreBreakdown(**defaults)


def test_relevance_base_prefers_rrf():
    assert _relevance_base(
        fusion_boost=0.03,
        bm25_score=20.0,
        semantic_score=0.8,
        bm25_max=20.0,
    ) == (0.03, "rrf")


def test_relevance_base_falls_back_to_normalized_bm25():
    assert _relevance_base(
        fusion_boost=0.0,
        bm25_score=5.0,
        semantic_score=0.0,
        bm25_max=10.0,
    ) == (0.5, "bm25")


def test_score_document_uses_rrf_as_primary_signal():
    hit = _score_document(
        "doc-1",
        query="rust async",
        query_tokens=["rust", "async"],
        bm25_score=6.0,
        semantic_score=0.5,
        bm25_rank=3,
        semantic_rank=8,
        fused_rank=2,
        bm25_max=10.0,
        fusion_boost=0.03,
        pagerank_boost=1.2,
        doc={
            "url": "https://example.com/rust-async",
            "title": "Async Rust guide",
            "published_at": "2026-06-19T00:00:00Z",
        },
    )

    assert hit is not None
    assert hit.breakdown.base_source == "rrf"
    assert hit.breakdown.base == 0.03
    assert hit.breakdown.final == hit.breakdown.base * hit.breakdown.metadata * hit.breakdown.title


def test_score_document_uses_semantic_fallback_without_rrf():
    hit = _score_document(
        "doc-1",
        query="embeddings",
        query_tokens=["embedding"],
        bm25_score=0.0,
        semantic_score=0.9,
        bm25_rank=None,
        semantic_rank=4,
        fused_rank=1,
        bm25_max=10.0,
        fusion_boost=0.0,
        pagerank_boost=1.0,
        doc={"url": "https://example.com/page", "title": "Page"},
    )

    assert hit is not None
    assert hit.breakdown.base_source == "sem"
    assert hit.breakdown.base == 0.9


def test_sort_hits_preserves_fused_rank_order():
    hits = [
        SearchHit("b", 0.05, _sample_breakdown(fused_rank=2, final=0.05)),
        SearchHit("a", 0.04, _sample_breakdown(fused_rank=1, final=0.04)),
        SearchHit("c", 0.06, _sample_breakdown(fused_rank=3, final=0.06)),
    ]

    _sort_hits(hits)

    assert [hit.doc_id for hit in hits] == ["a", "b", "c"]


def test_build_rank_maps():
    bm25_ranks, semantic_ranks, fused_ranks = build_rank_maps(
        ["a", "b", "c"],
        ["b", "a", "d"],
        {"a": 0.03, "b": 0.04, "c": 0.01, "d": 0.02},
    )

    assert bm25_ranks == {"a": 1, "b": 2, "c": 3}
    assert semantic_ranks == {"b": 1, "a": 2, "d": 3}
    assert fused_ranks == {"b": 1, "a": 2, "d": 3, "c": 4}


def test_format_score_breakdown_lists_ranks_and_contributors():
    text = format_score_breakdown(_sample_breakdown())

    assert "score 0.043" in text
    assert "bm25 #2 8.00" in text
    assert "sem #5 0.712" in text
    assert "fused #1" in text
    assert "rrf 0.0320" in text
    assert "recency×1.12" in text
    assert "fusion×" not in text


def test_format_score_breakdown_rich_splits_signals_and_multipliers():
    signals, multipliers = format_score_breakdown_rich(_sample_breakdown())

    assert signals.plain.startswith("score 0.043")
    assert "bm25 #2 8.00" in signals.plain
    assert "sem #5 0.712" in signals.plain
    assert "fused #1" in signals.plain
    assert "rrf 0.0320" in signals.plain
    assert multipliers is not None
    assert "recency×1.12" in multipliers.plain
    assert "title×1.25" in multipliers.plain
