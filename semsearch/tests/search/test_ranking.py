from datetime import datetime, timezone

from semsearch.search.ranking import apply_ranking, effective_timestamp, recency_boost


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
