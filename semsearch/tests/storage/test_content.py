import blake3

from semsearch.storage.content import (
    CONTENT_HASH_LEN,
    content_available,
    content_hash,
    save_content,
    try_read_content,
)
from semsearch.storage.content_models import init_content_db


def test_content_hash_is_blake3():
    html = "<html><body>Hello</body></html>"
    assert content_hash(html) == blake3.blake3(html.encode()).hexdigest()
    assert len(content_hash(html)) == CONTENT_HASH_LEN


def test_content_available_requires_valid_hash(tmp_path):
    init_content_db(tmp_path / "content.db")
    assert content_available("abc") is False


def test_try_read_content_returns_none_for_missing_blob(tmp_path):
    init_content_db(tmp_path / "content.db")
    assert try_read_content("0" * CONTENT_HASH_LEN) is None


def test_save_and_read_content_round_trip(tmp_path):
    init_content_db(tmp_path / "content.db")
    html = "<html><body>Hello</body></html>"
    digest = save_content(html)
    assert len(digest) == CONTENT_HASH_LEN
    assert content_available(digest)
    assert try_read_content(digest) == html
