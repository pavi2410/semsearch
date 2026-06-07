from html import unescape as unescape_html
from urllib.parse import unquote, urlparse

from rich import print as rprint
from rich.style import Style
from rich.text import Text

from Levenshtein import distance as lev_distance

from .core.search import search, get_docs

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


def _format_display_url(url: str) -> tuple[str, str]:
    link_url = unescape_html(url)
    display_url = unquote(link_url)
    return link_url, display_url


def _format_url_display(display_url: str, link_url: str) -> Text:
    hostname = urlparse(display_url).hostname or ""
    host_start = display_url.index(hostname)
    host_end = host_start + len(hostname)
    url_display = Text(
        display_url[:host_start]
    ).append(
        display_url[host_start:host_end], style=Style(italic=True, bold=True)
    ).append(
        display_url[host_end:]
    )
    url_display.stylize(Style(link=link_url))
    return url_display


def display_results(query: str, results: list[tuple[str, float]], query_time_ms: float, total_docs: int) -> None:
    docs = get_docs()
    rprint()
    rprint(f"Search results for [bold]{query}[/bold]")
    rprint(f"[dim]Found {len(results)} results from {total_docs} pages in {query_time_ms:.3f} ms[/dim]")
    rprint()

    for doc_id, score in results[:10]:
        doc = docs.get(doc_id, {})
        title = doc.get("title", "").strip() or "[italic]Untitled page[/italic]"
        link_url, display_url = _format_display_url(doc.get("url", ""))

        highlighted_title = Text.from_markup(title)
        score_str = f"({score:.2f})"

        url_display = _format_url_display(display_url, link_url)

        rprint(f"{highlighted_title} [dim]{score_str}[/dim]")
        rprint("\u21b3", url_display)
        rprint()


def main() -> None:
    import sys
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: semsearch <query>")
        return

    result = search(query)
    display_results(query, result.results, result.query_time_ms, result.total_docs)
