from datetime import datetime, timezone

from semsearch.search.ranking import (
    apply_ranking,
    compute_pagerank_boosts,
    effective_timestamp,
    https_boost,
    recency_boost,
)
from semsearch.storage.page import normalize_url


def test_effective_timestamp_prefers_modified_over_published():
    doc = {
        "modified_at": "2024-06-01T00:00:00Z",
        "published_at": "2024-01-01T00:00:00Z",
        "fetched_at": "2026-01-01T00:00:00Z",
    }
    assert effective_timestamp(doc) == datetime(2024, 6, 1, tzinfo=timezone.utc)


def test_effective_timestamp_falls_back_to_fetched_at():
    doc = {"fetched_at": "2026-01-15T12:00:00Z"}
    assert effective_timestamp(doc) == datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def test_recency_boost_is_higher_for_fresh_pages():
    now = datetime(2026, 6, 20, tzinfo=timezone.utc)
    fresh = recency_boost({"published_at": "2026-06-19T00:00:00Z"}, now=now)
    stale = recency_boost({"published_at": "2020-01-01T00:00:00Z"}, now=now)

    assert fresh > stale
    assert stale >= 0.85
    assert fresh <= 1.15


def test_apply_ranking_scales_bm25_by_recency():
    now = datetime(2026, 6, 20, tzinfo=timezone.utc)
    fresh_doc = {"published_at": "2026-06-19T00:00:00Z"}
    stale_doc = {"published_at": "2020-01-01T00:00:00Z"}

    fresh_score = apply_ranking(10.0, fresh_doc)
    stale_score = apply_ranking(10.0, stale_doc)

    assert fresh_score > stale_score


def test_https_boost_prefers_secure_urls():
    assert https_boost({"url": "https://example.com"}) > https_boost({"url": "http://example.com"})


def test_apply_ranking_includes_https_boost():
    secure = apply_ranking(10.0, {"url": "https://example.com"})
    insecure = apply_ranking(10.0, {"url": "http://example.com"})
    assert secure > insecure


def test_pagerank_boosts_linked_pages():
    docs = {
        "a": {"url": "https://example.com/a"},
        "b": {"url": "https://example.com/b"},
        "c": {"url": "https://example.com/c"},
    }
    url_to_doc = {normalize_url(doc["url"]): doc_id for doc_id, doc in docs.items()}
    links = [
        ("a", "https://example.com/b"),
        ("b", "https://example.com/c"),
    ]
    boosts = compute_pagerank_boosts(list(docs), url_to_doc, links)

    assert boosts["c"] > boosts["b"] > boosts["a"]
    assert boosts["a"] == 1.0
    assert boosts["c"] == 1.3


def test_pagerank_uses_boost_only_floor():
    boosts = compute_pagerank_boosts(["a", "b", "c"], {}, [])
    assert all(boost >= 1.0 for boost in boosts.values())


def test_pagerank_returns_neutral_boost_without_links():
    boosts = compute_pagerank_boosts(["a", "b"], {}, [])
    assert boosts == {"a": 1.0, "b": 1.0}


def test_apply_ranking_includes_pagerank_boost():
    doc = {"url": "https://example.com"}
    base = apply_ranking(10.0, doc, pagerank_boost=1.0)
    boosted = apply_ranking(10.0, doc, pagerank_boost=1.1)
    assert boosted > base
