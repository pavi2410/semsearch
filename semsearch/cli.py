from html import unescape as unescape_html
from urllib.parse import unquote, urlparse

from rich import print as rprint
from rich.style import Style
from rich.text import Text

from .core.search import search, get_docs

HIGHLIGHT_STYLE = Style(bgcolor="yellow")


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

    for doc_id, score in results[:3]:
        doc = docs.get(doc_id, {})
        title = doc.get("title", "").strip() or "Untitled page"
        link_url, display_url = _format_display_url(doc.get("url", ""))

        highlighted_title = Text(title)
        highlighted_title.highlight_words(query.split(), HIGHLIGHT_STYLE, case_sensitive=False)
        score_str = f"({score:.4f})"

        url_display = _format_url_display(display_url, link_url)
        url_display.highlight_words(query.split(), HIGHLIGHT_STYLE, case_sensitive=False)

        rprint(highlighted_title, f"[dim]{score_str}[/dim]")
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
