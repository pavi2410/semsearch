def make_snippet(text: str, query: str, max_len: int = 160) -> str:
    """Return a short excerpt from text, biased toward query term matches."""
    text = text.strip()
    if not text:
        return ""

    if len(text) <= max_len:
        return text

    terms = [term.lower() for term in query.split() if term.strip()]
    lower = text.lower()
    match_pos = -1
    for term in terms:
        pos = lower.find(term)
        if pos != -1 and (match_pos == -1 or pos < match_pos):
            match_pos = pos

    if match_pos == -1:
        start = 0
        snippet = text[:max_len]
    else:
        start = max(0, match_pos - max_len // 3)
        snippet = text[start : start + max_len]

    if start > 0:
        snippet = "…" + snippet
    if start + len(snippet) < len(text):
        snippet = snippet.rstrip() + "…"
    return snippet.strip()
