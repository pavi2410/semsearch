from semsearch.search.dedup import dedupe_results
from semsearch.storage.page import canonical_key, normalize_url


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
        ("variant", 9.0),
        ("canonical", 8.0),
        ("other", 7.0),
    ]

    deduped = dedupe_results(results, docs)

    assert deduped == [("variant", 9.0), ("other", 7.0)]
