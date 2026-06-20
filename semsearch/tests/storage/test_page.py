from semsearch.search.dedup import dedupe_results
from semsearch.search.search import ScoreBreakdown, SearchHit
from semsearch.storage.page import canonical_key, normalize_url


def _hit(doc_id: str, score: float) -> SearchHit:
    breakdown = ScoreBreakdown(
        bm25=score,
        semantic=0.0,
        bm25_rank=1,
        semantic_rank=None,
        fused_rank=1,
        base=score,
        base_source="bm25",
        recency=1.0,
        https=1.0,
        pagerank=1.0,
        metadata=1.0,
        title=1.0,
        fusion=0.0,
        fusion_multiplier=1.0,
        final=score,
    )
    return SearchHit(doc_id=doc_id, score=score, breakdown=breakdown)


def test_normalize_url_strips_trailing_slash():
    assert normalize_url("https://Example.com/path/") == "https://example.com/path"


def test_normalize_url_strips_locale_query_params():
    assert (
        normalize_url("https://github.com/security/advanced-security?locale=en-US")
        == "https://github.com/security/advanced-security"
    )


def test_normalize_url_preserves_non_locale_query_params():
    assert (
        normalize_url("https://example.com/page?id=42&lang=en")
        == "https://example.com/page?id=42"
    )


def test_canonical_key_prefers_canonical_url():
    key = canonical_key(
        "https://example.com/a?locale=en-US",
        "https://example.com/canonical/",
    )
    assert key == "https://example.com/canonical"


def test_dedupe_results_keeps_highest_score_per_canonical_url():
    docs = {
        "variant": {
            "url": "https://example.com/page?locale=en-US",
            "canonical_url": "https://example.com/page",
        },
        "canonical": {
            "url": "https://example.com/page",
            "canonical_url": "https://example.com/page",
        },
        "other": {
            "url": "https://example.com/other",
            "canonical_url": "",
        },
    }
    results = [
        _hit("variant", 9.0),
        _hit("canonical", 8.0),
        _hit("other", 7.0),
    ]

    deduped = dedupe_results(results, docs)

    assert [hit.doc_id for hit in deduped] == ["variant", "other"]
