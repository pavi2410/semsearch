from rich.text import Text

from semsearch.cli import HIGHLIGHT_STYLE, _apply_query_highlights, _highlight_text


def _highlighted_parts(text: Text) -> list[str]:
    return [text.plain[span.start : span.end] for span in text.spans if span.style == HIGHLIGHT_STYLE]


def test_short_terms_highlight_whole_words_only():
    parts = _highlighted_parts(_highlight_text("and an company", "an"))

    assert parts == ["an"]
    assert "and" not in parts
    assert "company" not in parts


def test_long_terms_still_highlight_substrings():
    parts = _highlighted_parts(_highlight_text("startup startups", "startup"))

    assert parts == ["startup", "startup"]
