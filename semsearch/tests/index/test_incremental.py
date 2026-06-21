import pytest

from semsearch.index.indexer import (
    _process_page,
    build_index_stats,
    filter_pages_with_content,
    plan_index,
)
from semsearch.storage.models import Page, TokenCache, init_db
from semsearch.storage.token_cache import load_tokens, save_tokens


def test_process_page_uses_stored_url_hash(monkeypatch):
    html = "<html><head><title>Hi</title></head><body><p>Hello</p></body></html>"
    meta = {
        "url": "https://example.com/page?locale=en-US",
        "urlHash": "stored-hash",
        "contentHash": "abc123",
    }
    monkeypatch.setattr("semsearch.index.indexer.try_read_content", lambda _hash: html)
    monkeypatch.setattr("semsearch.index.indexer.is_indexable_page", lambda *_args: True)
    monkeypatch.setattr("semsearch.index.indexer.preprocess", lambda text: text.split())
    monkeypatch.setattr(
        "semsearch.index.indexer.extract_page_metadata",
        lambda _html, _url: type(
            "Meta",
            (),
            {
                "title": "Hi",
                "description": "",
                "body_text": "Hello",
                "outbound_links": [],
            },
        )(),
    )
    monkeypatch.setattr(
        "semsearch.storage.page.url_hash",
        lambda _url: "recomputed-hash",
    )

    result = _process_page(meta)

    assert result is not None
    _url, doc_id, _content_hash, _page_meta, _tokens = result
    assert doc_id == "stored-hash"


def test_plan_index_reuses_unchanged_pages():
    pages = [
        {
            "urlHash": "aaa",
            "contentHash": "hash-a",
            "indexedContentHash": "hash-a",
        },
        {
            "urlHash": "bbb",
            "contentHash": "hash-b",
            "indexedContentHash": "",
        },
    ]
    tokens = {"hash-a": ["hello", "world"]}

    plan = plan_index(pages, force=False, token_loader=tokens.get)

    assert plan.reused == {"aaa": ["hello", "world"]}
    assert len(plan.to_process) == 1
    assert plan.to_process[0]["urlHash"] == "bbb"


def test_filter_pages_with_content_drops_missing_files():
    pages = [
        {"urlHash": "aaa", "contentHash": "0123456789abcdef"},
        {"urlHash": "bbb", "contentHash": "abc"},
    ]
    valid, missing = filter_pages_with_content(pages)
    assert valid == []
    assert missing == 2


def test_plan_index_force_reprocesses_all():
    pages = [
        {
            "urlHash": "aaa",
            "contentHash": "hash-a",
            "indexedContentHash": "hash-a",
        }
    ]

    plan = plan_index(pages, force=True, token_loader=lambda _hash: ["cached"])

    assert plan.reused == {}
    assert len(plan.to_process) == 1


def test_build_index_stats_counts_removed_pages():
    pages = [{"urlHash": "aaa", "contentHash": "hash-a", "indexedContentHash": "hash-a"}]
    plan = plan_index(pages, force=False, token_loader=lambda _hash: ["token"])

    stats = build_index_stats(
        pages,
        plan,
        previous_doc_ids=["aaa", "gone"],
        skipped=0,
    )

    assert stats.reused == 1
    assert stats.removed == 1


@pytest.fixture
def db(tmp_path):
    init_db(tmp_path / "test.db")
    yield


def test_token_cache_round_trip(db):
    save_tokens("abc123", ["one", "two"])
    assert load_tokens("abc123") == ["one", "two"]
    assert load_tokens("missing") is None


def test_persist_page_sets_indexed_content_hash(db):
    Page.replace(
        url_hash="doc1",
        url="https://example.com",
        fetched_at="2026-01-01T00:00:00Z",
        content_hash="hash-a",
    ).execute()

    Page.update(indexed_content_hash="hash-a").where(Page.url_hash == "doc1").execute()
    page = Page.get_by_id("doc1")
    assert page.indexed_content_hash == "hash-a"

    save_tokens("hash-a", ["hello"])
    row = (
        TokenCache.select(TokenCache.tokens.json())
        .where(TokenCache.content_hash == "hash-a")
        .get()
    )
    assert row.tokens == ["hello"]
