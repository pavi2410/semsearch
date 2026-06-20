from html import unescape as unescape_html
import re
from urllib.parse import unquote, urlparse

from rich import print as rprint
from rich.console import Group
from rich.rule import Rule
from rich.style import Style
from rich.text import Text

from .search.search import SearchHit, format_score_breakdown_rich, get_docs, search
from .search.snippet import make_snippet

HIGHLIGHT_STYLE = Style(bgcolor="yellow")
RESULT_LIMIT = 10
_MIN_SUBSTRING_HIGHLIGHT_LEN = 3


def _format_display_url(url: str) -> tuple[str, str]:
    link_url = unescape_html(url)
    display_url = unquote(link_url)
    return link_url, display_url


def _format_url_display(display_url: str, link_url: str) -> Text:
    hostname = urlparse(display_url).hostname or ""
    host_start = display_url.index(hostname)
    host_end = host_start + len(hostname)
    url_display = (
        Text(display_url[:host_start], style=Style(dim=True))
        .append(display_url[host_start:host_end], style=Style(italic=True, bold=True, dim=False))
        .append(display_url[host_end:], style=Style(dim=True))
    )
    url_display.stylize(Style(link=link_url))
    return url_display


def _apply_query_highlights(text: Text, query: str) -> Text:
    for term in query.split():
        cleaned = term.strip()
        if not cleaned:
            continue
        if len(cleaned) < _MIN_SUBSTRING_HIGHLIGHT_LEN:
            pattern = rf"(?i)\b{re.escape(cleaned)}\b"
            text.highlight_regex(pattern, HIGHLIGHT_STYLE)
        else:
            text.highlight_words([cleaned], HIGHLIGHT_STYLE, case_sensitive=False)
    return text


def _highlight_text(text: str, query: str) -> Text:
    return _apply_query_highlights(Text(text), query)


def _result_snippet(doc: dict[str, str], query: str) -> str:
    description = doc.get("description", "").strip()
    body_excerpt = doc.get("body_excerpt", "").strip()
    source = body_excerpt or description
    if description and _looks_like_nav_chrome(body_excerpt):
        source = description
    return make_snippet(source, query)


def _looks_like_nav_chrome(text: str) -> bool:
    lowered = text.lower()
    nav_markers = ("open menu", "skip to content", "sign in", "log in")
    hits = sum(1 for marker in nav_markers if marker in lowered)
    return hits >= 2 or (hits == 1 and len(text.split()) > 12)


def _format_score_lines(hit: SearchHit, *, language: str) -> Group:
    signals, multipliers = format_score_breakdown_rich(hit.breakdown)
    indent = "    "
    lines: list[Text] = [Text(indent, style="dim") + signals]
    if multipliers is not None:
        lines.append(Text(indent, style="dim") + multipliers)
    if language:
        lines[-1].append(f" [{language}]", style="dim italic")
    return Group(*lines)


def display_results(
    query: str, results: list[SearchHit], query_time_ms: float, total_docs: int
) -> None:
    docs = get_docs()
    shown = results[:RESULT_LIMIT]

    rprint()
    rprint(Rule(f"[bold] {query} [/bold]", style="dim"))
    rprint(
        f"[dim]{len(results)} results from {total_docs:,} pages · {query_time_ms:.1f} ms[/dim]"
    )
    rprint()

    for rank, hit in enumerate(shown, start=1):
        doc = docs.get(hit.doc_id, {})
        title = doc.get("title", "").strip() or "Untitled page"
        link_url, display_url = _format_display_url(doc.get("url", ""))
        snippet = _result_snippet(doc, query)
        language = doc.get("language", "").strip()

        highlighted_title = _highlight_text(title, query)
        highlighted_title.stylize("bold")

        url_display = _format_url_display(display_url, link_url)
        _apply_query_highlights(url_display, query)

        rprint(Text(f"{rank:>2}. ", style="dim"), highlighted_title)
        rprint(Text("    ↳ ", style="dim"), url_display)
        if snippet:
            snippet_text = _highlight_text(snippet, query)
            snippet_text.stylize("dim")
            rprint(Text("    ", style="dim"), snippet_text)
        print()
        rprint(_format_score_lines(hit, language=language))

        if rank < len(shown):
            rprint(Rule(style="dim bright_black"))

    rprint()


def main() -> None:
    import sys

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: semsearch <query>")
        return

    result = search(query)
    display_results(query, result.results, result.query_time_ms, result.total_docs)
