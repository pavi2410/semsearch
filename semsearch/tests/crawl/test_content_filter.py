import json

from semsearch.crawl.content_filter import (
    is_fetchable_document_url,
    is_html_content_type,
    is_indexable_page,
    looks_like_json,
    url_path_extension,
)


def test_url_path_extension():
    assert url_path_extension("https://example.com/item.json?print=pretty") == ".json"
    assert url_path_extension("https://example.com/page") == ""


def test_is_fetchable_document_url_rejects_json():
    assert is_fetchable_document_url("https://example.com/page") is True
    assert (
        is_fetchable_document_url(
            "https://hacker-news.firebaseio.com/v0/item/8863.json?print=pretty"
        )
        is False
    )


def test_is_html_content_type():
    assert is_html_content_type("text/html; charset=utf-8") is True
    assert is_html_content_type("application/json") is False


def test_looks_like_json():
    assert looks_like_json('{"id": 1, "title": "hello"}') is True
    assert looks_like_json("<html><body>hello</body></html>") is False


def test_is_indexable_page():
    assert is_indexable_page(
        "https://example.com/page",
        "<html><body>Article text</body></html>",
        "text/html",
    )
    assert is_indexable_page(
        "https://example.com/item.json",
        json.dumps({"title": "Poll"}),
        "application/json",
    ) is False
