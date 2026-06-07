from urllib.parse import urlparse

from rich import print as rprint
from rich.style import Style
from rich.text import Text

from Levenshtein import distance as lev_distance

from .search import search, get_docs

HIGHLIGHT_STYLE = "bold"


def _highlight_span(text: str, spans: list[tuple[int, int]]) -> str:
    result = Text()
    prev = 0
    for start, end in spans:
        if start > prev:
            result.append(text[prev:start])
        result.append(text[start:end], style=HIGHLIGHT_STYLE)
        prev = end
    if prev < len(text):
        result.append(text[prev:])
    return result  # type: ignore[return-value]


def _add_highlights(text: str, query: str) -> str:
    query_tokens = query.lower().split()
    if not query_tokens:
        return text

    text_lower = text.lower()
    spans: list[tuple[int, int]] = []

    for qt in query_tokens:
        for i in range(len(text_lower) - len(qt) + 1):
            candidate = text_lower[i:i + len(qt)]
            if lev_distance(candidate, qt) < len(qt) / 2:
                spans.append((i, i + len(qt)))

    spans.sort(key=lambda x: (x[0], x[1]))

    merged: list[tuple[int, int]] = []
    for span in spans:
        if not merged:
            merged.append(span)
        else:
            last = merged[-1]
            if span[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], span[1]))
            else:
                merged.append(span)

    return str(_highlight_span(text, merged))


def display_results(query: str, results: list[tuple[str, float]]) -> None:
    docs = get_docs()
    rprint()
    rprint(f"Search results for [bold]{query}[/bold]")
    rprint(f"[dim]Found {len(results)} results[/dim]")
    rprint()

    for doc_id, score in results[:10]:
        doc = docs.get(doc_id, {})
        title = doc.get("title", "").strip() or "[italic]Untitled page[/italic]"
        url = doc.get("url", "")

        highlighted_title = Text.from_markup(title)

        hostname = urlparse(url).hostname or ""
        host_start = url.index(hostname)
        host_end = host_start + len(hostname)
        url_display = Text(
            url[:host_start]
        ).append(
            url[host_start:host_end], style=Style(italic=True, underline=True)
        ).append(
            url[host_end:]
        )

        score_str = f"({score:.2f})"

        rprint(f"{highlighted_title} [dim]{score_str}[/dim]")
        rprint(f"\u21b3 {url_display}")
        rprint()


def main() -> None:
    import sys
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: semsearch <query>")
        return

    result = search(query)
    display_results(query, result.results)
