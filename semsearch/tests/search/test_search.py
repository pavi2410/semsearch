from semsearch.search.search import (
    ScoreBreakdown,
    _score_document,
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
        "base": 8.0,
        "base_source": "bm25",
        "recency": 1.12,
        "https": 1.05,
        "pagerank": 1.15,
        "metadata": 1.08,
        "title": 1.25,
        "fusion": 0.032,
        "fusion_multiplier": 1.032,
        "final": 12.345,
    }
    defaults.update(overrides)
    return ScoreBreakdown(**defaults)


def test_score_document_combines_signals():
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
    assert hit.score == hit.breakdown.final
    assert hit.breakdown.base_source == "bm25"
    assert hit.breakdown.bm25_rank == 3
    assert hit.breakdown.semantic_rank == 8
    assert hit.breakdown.fused_rank == 2
    assert hit.breakdown.title == 1.10
    assert hit.breakdown.fusion_multiplier > 1.0


def test_score_document_uses_semantic_base_when_stronger():
    hit = _score_document(
        "doc-1",
        query="embeddings",
        query_tokens=["embedding"],
        bm25_score=1.0,
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
    assert hit.breakdown.base == 0.9 * 8.0


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

    assert "score 12.345" in text
    assert "bm25 #2 8.00" in text
    assert "sem #5 0.712" in text
    assert "fused #1" in text
    assert "rrf 0.0320" in text
    assert "recency×1.12" in text
    assert "fusion×1.032" in text
    assert "base bm25" not in text


def test_format_score_breakdown_rich_splits_signals_and_multipliers():
    signals, multipliers = format_score_breakdown_rich(_sample_breakdown())

    assert signals.plain.startswith("score 12.345")
    assert "bm25 #2 8.00" in signals.plain
    assert "sem #5 0.712" in signals.plain
    assert "fused #1" in signals.plain
    assert "rrf 0.0320" in signals.plain
    assert multipliers is not None
    assert "recency×1.12" in multipliers.plain
    assert "title×1.25" in multipliers.plain
