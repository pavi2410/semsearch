from semsearch.storage.content import (
    CONTENT_DIR,
    content_available,
    save_content,
    try_read_content,
)


def test_content_available_requires_full_hash(tmp_path, monkeypatch):
    monkeypatch.setattr("semsearch.storage.content.CONTENT_DIR", tmp_path)
    assert content_available("abc") is False


def test_try_read_content_returns_none_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("semsearch.storage.content.CONTENT_DIR", tmp_path)
    assert try_read_content("0123456789abcdef") is None


def test_save_and_read_content_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("semsearch.storage.content.CONTENT_DIR", tmp_path)
    content_hash = save_content("<html><body>Hello</body></html>")
    assert content_available(content_hash)
    assert try_read_content(content_hash) == "<html><body>Hello</body></html>"
    assert (tmp_path / content_hash[:2] / f"{content_hash[2:]}.html.gz").is_file()
