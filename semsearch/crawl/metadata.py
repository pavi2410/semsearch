import html as html_mod
import json
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, SoupStrainer, XMLParsedAsHTMLWarning

from .content_filter import is_fetchable_document_url
from .main_content import extract_main_text

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_HEAD = SoupStrainer("head")
_BODY = SoupStrainer("body")
_A_TAG = SoupStrainer("a", href=True)

_ARTICLE_TYPES = {
    "article",
    "newsarticle",
    "blogposting",
    "webpage",
    "report",
    "scholarlyarticle",
}
_ISO8601 = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
_EXCERPT_LEN = 500


@dataclass
class PageMetadata:
    title: str = ""
    description: str = ""
    canonical_url: str = ""
    og_title: str = ""
    og_description: str = ""
    published_at: str = ""
    modified_at: str = ""
    body_text: str = ""
    body_excerpt: str = ""
    outbound_links: list[str] = field(default_factory=list)
    jsonld_types: list[str] = field(default_factory=list)


def _unescape(text: str) -> str:
    return html_mod.unescape(text.strip()) if text else ""


def _normalize_iso8601(value: str) -> str:
    value = value.strip()
    if not value or not _ISO8601.match(value):
        return ""
    try:
        normalized = value.replace("Z", "+00:00")
        if "T" not in normalized and " " not in normalized:
            normalized = f"{normalized}T00:00:00+00:00"
        dt = datetime.fromisoformat(normalized.replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return ""


def _pick(*values: str) -> str:
    for value in values:
        cleaned = _unescape(value)
        if cleaned:
            return cleaned
    return ""


def _meta_content(tag) -> str:
    if tag.name == "meta":
        return _unescape(tag.get("content", ""))
    if tag.name == "link":
        return _unescape(tag.get("href", ""))
    return ""


def _parse_head_tags(head) -> dict[str, str]:
    fields = {
        "description": "",
        "og_title": "",
        "og_description": "",
        "published_at": "",
        "modified_at": "",
        "canonical_url": "",
    }
    for tag in head.find_all(["meta", "link"]):
        name = (tag.get("name") or "").lower()
        prop = (tag.get("property") or "").lower()
        rel = (tag.get("rel") or [])
        if isinstance(rel, str):
            rel = [rel]
        rel = [r.lower() for r in rel]
        content = _meta_content(tag)

        if name == "description" and not fields["description"]:
            fields["description"] = content
        elif prop == "og:title" and not fields["og_title"]:
            fields["og_title"] = content
        elif prop == "og:description" and not fields["og_description"]:
            fields["og_description"] = content
        elif prop in {"article:published_time", "og:article:published_time"}:
            fields["published_at"] = _normalize_iso8601(content) or fields["published_at"]
        elif prop in {"article:modified_time", "og:article:modified_time", "og:updated_time"}:
            fields["modified_at"] = _normalize_iso8601(content) or fields["modified_at"]
        elif "canonical" in rel and not fields["canonical_url"]:
            fields["canonical_url"] = content

    return fields


def _jsonld_objects(raw) -> list[dict]:
    if isinstance(raw, list):
        objects: list[dict] = []
        for item in raw:
            objects.extend(_jsonld_objects(item))
        return objects
    if not isinstance(raw, dict):
        return []

    if "@graph" in raw:
        return _jsonld_objects(raw["@graph"])

    return [raw]


def _jsonld_types(node: dict) -> set[str]:
    raw_type = node.get("@type", "")
    if isinstance(raw_type, list):
        return {t.lower() for t in raw_type if isinstance(t, str)}
    if isinstance(raw_type, str):
        return {raw_type.split("/")[-1].lower()}
    return set()


def _parse_jsonld(head) -> dict[str, str]:
    fields = {
        "title": "",
        "description": "",
        "published_at": "",
        "modified_at": "",
        "canonical_url": "",
    }
    types: list[str] = []

    for script in head.find_all("script", attrs={"type": "application/ld+json"}):
        if not script.string:
            continue
        try:
            payload = json.loads(script.string)
        except json.JSONDecodeError:
            continue

        for node in _jsonld_objects(payload):
            node_types = _jsonld_types(node)
            if not node_types & _ARTICLE_TYPES:
                continue

            types.extend(sorted(node_types & _ARTICLE_TYPES))
            fields["title"] = fields["title"] or _pick(
                str(node.get("headline", "")),
                str(node.get("name", "")),
            )
            fields["description"] = fields["description"] or _pick(
                str(node.get("description", "")),
            )
            fields["published_at"] = fields["published_at"] or _normalize_iso8601(
                str(node.get("datePublished", ""))
            )
            fields["modified_at"] = fields["modified_at"] or _normalize_iso8601(
                str(node.get("dateModified", ""))
            )
            url = str(node.get("url", "") or node.get("@id", ""))
            if url.startswith("http"):
                fields["canonical_url"] = fields["canonical_url"] or url

    return {**fields, "jsonld_types": ",".join(dict.fromkeys(types))}


def _extract_body_text(html: str) -> str:
    main_text = extract_main_text(html)
    if main_text:
        return main_text

    body = BeautifulSoup(html, "lxml", parse_only=_BODY)
    for tag in body.find_all(["script", "style", "noscript"]):
        tag.decompose()
    return _unescape(body.get_text(separator=" ", strip=True))


def _extract_title(head) -> str:
    title_tag = head.find("title")
    if title_tag and title_tag.string:
        return _unescape(title_tag.string)
    return ""


def extract_outbound_links(html: str, base_url: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for a in BeautifulSoup(html, "lxml", parse_only=_A_TAG):
        href = a.get("href")
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        normalized = parsed._replace(fragment="").geturl()
        if normalized not in seen and is_fetchable_document_url(normalized):
            seen.add(normalized)
            links.append(normalized)
    return links


def extract_page_metadata(html: str, page_url: str) -> PageMetadata:
    soup = BeautifulSoup(html, "lxml", parse_only=SoupStrainer(["head", "body"]))
    head = soup.find("head") or soup
    head_fields = _parse_head_tags(head)
    jsonld_fields = _parse_jsonld(head)
    body_text = _extract_body_text(html)

    title = _pick(head_fields.get("og_title", ""), _extract_title(head), jsonld_fields["title"])
    description = _pick(
        head_fields.get("og_description", ""),
        head_fields.get("description", ""),
        jsonld_fields["description"],
    )
    if not description and body_text:
        description = body_text[:160]
        if len(body_text) > 160:
            description = description.rsplit(" ", 1)[0] + "…"

    published_at = _pick(head_fields["published_at"], jsonld_fields["published_at"])
    modified_at = _pick(head_fields["modified_at"], jsonld_fields["modified_at"])
    canonical_url = _pick(head_fields["canonical_url"], jsonld_fields["canonical_url"])

    excerpt = body_text[:_EXCERPT_LEN]
    if len(body_text) > _EXCERPT_LEN:
        excerpt = excerpt.rsplit(" ", 1)[0] + "…"

    jsonld_types = [
        t.strip()
        for t in jsonld_fields.get("jsonld_types", "").split(",")
        if t.strip()
    ]

    return PageMetadata(
        title=title,
        description=description,
        canonical_url=canonical_url,
        og_title=head_fields.get("og_title", ""),
        og_description=head_fields.get("og_description", ""),
        published_at=published_at,
        modified_at=modified_at,
        body_text=body_text,
        body_excerpt=excerpt,
        outbound_links=extract_outbound_links(html, page_url),
        jsonld_types=jsonld_types,
    )
