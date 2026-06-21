from semsearch.storage.models import Link, TargetUrl, init_db
from semsearch.storage.url_intern import intern_urls


def test_intern_urls_assigns_stable_ids(tmp_path):
    init_db(tmp_path / "test.db")

    first = intern_urls(["https://example.com/a", "https://example.com/b"])
    second = intern_urls(["https://example.com/b", "https://example.com/c"])

    assert first == {
        "https://example.com/a": 1,
        "https://example.com/b": 2,
    }
    assert second["https://example.com/b"] == 2
    assert second["https://example.com/c"] == 3
    assert TargetUrl.select().count() == 3


def test_intern_urls_normalizes_before_insert(tmp_path):
    init_db(tmp_path / "test.db")

    mapping = intern_urls(["https://Example.com/path/?locale=en-US"])

    assert mapping == {"https://example.com/path": 1}


def test_intern_urls_deduplicates_within_call(tmp_path):
    init_db(tmp_path / "test.db")

    mapping = intern_urls(
        [
            "https://example.com/a",
            "https://example.com/a",
            "https://example.com/b",
        ]
    )

    assert mapping == {
        "https://example.com/a": 1,
        "https://example.com/b": 2,
    }
    assert TargetUrl.select().count() == 2


def test_intern_urls_empty_input(tmp_path):
    init_db(tmp_path / "test.db")
    assert intern_urls([]) == {}


def test_save_links_uses_interned_target_ids(tmp_path):
    from semsearch.index.indexer import _save_links

    init_db(tmp_path / "test.db")

    _save_links(
        "source-doc",
        [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/a",
        ],
    )

    rows = list(
        Link.select(Link.source_hash, TargetUrl.url)
        .join(TargetUrl, on=(Link.target_id == TargetUrl.id))
        .order_by(TargetUrl.url)
        .tuples()
    )
    assert rows == [
        ("source-doc", "https://example.com/a"),
        ("source-doc", "https://example.com/b"),
    ]
