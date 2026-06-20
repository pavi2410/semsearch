from semsearch.search.search import ScoreBreakdown, _score_document, format_score_breakdown


def test_score_document_combines_signals():
    hit = _score_document(
        "doc-1",
        query="rust async",
        query_tokens=["rust", "async"],
        bm25_score=6.0,
        semantic_score=0.5,
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
    assert hit.breakdown.title == 1.10
    assert hit.breakdown.fusion_multiplier > 1.0


def test_score_document_uses_semantic_base_when_stronger():
    hit = _score_document(
        "doc-1",
        query="embeddings",
        query_tokens=["embedding"],
        bm25_score=1.0,
        semantic_score=0.9,
        bm25_max=10.0,
        fusion_boost=0.0,
        pagerank_boost=1.0,
        doc={"url": "https://example.com/page", "title": "Page"},
    )

    assert hit is not None
    assert hit.breakdown.base_source == "sem"
    assert hit.breakdown.base == 0.9 * 8.0


def test_format_score_breakdown_lists_contributors():
    text = format_score_breakdown(
        ScoreBreakdown(
            bm25=8.0,
            semantic=0.712,
            base=8.0,
            base_source="bm25",
            recency=1.12,
            https=1.05,
            pagerank=1.15,
            metadata=1.08,
            title=1.25,
            fusion=0.032,
            fusion_multiplier=1.032,
            final=12.345,
        )
    )

    assert "score 12.345" in text
    assert "bm25 8.00" in text
    assert "sem 0.712" in text
    assert "base bm25" in text
    assert "recency×1.12" in text
    assert "https×1.05" in text
    assert "pagerank×1.15" in text
    assert "title×1.25" in text
    assert "fusion×1.032" in text
