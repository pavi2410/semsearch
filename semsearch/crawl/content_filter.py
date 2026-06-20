import json
from urllib.parse import urlparse

_NON_HTML_EXTENSIONS = {
    ".json",
    ".xml",
    ".pdf",
    ".zip",
    ".gz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".css",
    ".js",
    ".woff",
    ".woff2",
    ".ttf",
    ".mp3",
    ".mp4",
    ".webm",
}

_HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}


def url_path_extension(url: str) -> str:
    path = urlparse(url).path.lower()
    filename = path.rsplit("/", 1)[-1]
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1]


def is_fetchable_document_url(url: str) -> bool:
    """Return False for obvious non-HTML resources such as JSON API endpoints."""
    return url_path_extension(url) not in _NON_HTML_EXTENSIONS


def is_html_content_type(content_type: str | None) -> bool:
    if not content_type:
        return True
    mime = content_type.split(";", 1)[0].strip().lower()
    return mime in _HTML_CONTENT_TYPES


def looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")):
        return False
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return True


def is_indexable_page(url: str, content: str, content_type: str | None = None) -> bool:
    if not is_fetchable_document_url(url):
        return False
    if content_type is not None and not is_html_content_type(content_type):
        return False
    return not looks_like_json(content)
