from semsearch.search.snippet import make_snippet


def test_make_snippet_returns_short_text_unchanged():
    assert make_snippet("short text", "query") == "short text"


def test_make_snippet_adds_ellipsis_for_long_text():
    text = "word " * 50
    snippet = make_snippet(text, "missing-term", max_len=20)
    assert snippet.endswith("…")
