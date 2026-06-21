import hashlib

from semsearch.storage.content import (
    CONTENT_DIR,
    CONTENT_HASH_LEN,
    content_available,
    content_hash,
    save_content,
    try_read_content,
)


def test_content_hash_is_full_sha256():
    html = "<html><body>Hello</body></html>"
    assert content_hash(html) == hashlib.sha256(html.encode()).hexdigest()
    assert len(content_hash(html)) == CONTENT_HASH_LEN


def test_content_available_requires_valid_hash(tmp_path, monkeypatch):
    monkeypatch.setattr("semsearch.storage.content.CONTENT_DIR", tmp_path)
    assert content_available("abc") is False


def test_try_read_content_returns_none_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("semsearch.storage.content.CONTENT_DIR", tmp_path)
    assert try_read_content("0" * CONTENT_HASH_LEN) is None


def test_save_and_read_content_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("semsearch.storage.content.CONTENT_DIR", tmp_path)
    html = "<html><body>Hello</body></html>"
    digest = save_content(html)
    assert len(digest) == CONTENT_HASH_LEN
    assert content_available(digest)
    assert try_read_content(digest) == html
    assert (tmp_path / digest[:2] / f"{digest[2:]}.html.gz").is_file()
