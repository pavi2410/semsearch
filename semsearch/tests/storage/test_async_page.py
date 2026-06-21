from semsearch.storage.models import init_db
from semsearch.storage.page import read_page_meta, save_page


def test_save_and_read_page_meta(tmp_path):
    init_db(tmp_path / "test.db")
    saved = save_page(
        "https://example.com/page",
        "<html><body>hello</body></html>",
        "2024-01-01T00:00:00Z",
        etag='"abc"',
        http_last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
    )

    meta = read_page_meta("https://example.com/page")

    assert saved["url"] == "https://example.com/page"
    assert meta is not None
    assert meta["url"] == "https://example.com/page"
    assert meta["lastFetchedAt"] == "2024-01-01T00:00:00Z"
    assert meta["etag"] == '"abc"'
    assert meta["httpLastModified"] == "Mon, 01 Jan 2024 00:00:00 GMT"
    assert meta["contentHash"]


def test_read_page_meta_missing(tmp_path):
    init_db(tmp_path / "test.db")
    assert read_page_meta("https://example.com/missing") is None
