import json
import re

from bs4 import BeautifulSoup, SoupStrainer
from langdetect import LangDetectException, detect_langs

_HEAD = SoupStrainer("head")
_LOCALE_RE = re.compile(r"^([a-z]{2})(?:[-_][A-Za-z]{2})?$")
_HTML_LANG_RE = re.compile(r"""<html[^>]*\slang=['"]([^'"]+)['"]""", re.I)
_MIN_TEXT_CHARS = 40
_MIN_DETECT_CONFIDENCE = 0.85


def normalize_language_code(value: str) -> str:
    code = value.strip().lower().replace("_", "-")
    if not code:
        return ""
    code = code.split(",")[0].strip()
    match = _LOCALE_RE.match(code)
    if match:
        return match.group(1)
    return ""


def _language_from_html_tag(html: str) -> str:
    match = _HTML_LANG_RE.search(html)
    if match:
        return normalize_language_code(match.group(1))
    return ""


def _language_from_head(head) -> str:
    for tag in head.find_all(["meta", "link"]):
        if tag.name == "meta":
            http_equiv = (tag.get("http-equiv") or "").lower()
            name = (tag.get("name") or "").lower()
            prop = (tag.get("property") or "").lower()
            content = tag.get("content") or ""
            if http_equiv == "content-language" or name == "language":
                lang = normalize_language_code(content)
                if lang:
                    return lang
            if prop == "og:locale":
                lang = normalize_language_code(content)
                if lang:
                    return lang
    return ""


def _language_from_jsonld(head) -> str:
    for script in head.find_all("script", attrs={"type": "application/ld+json"}):
        if not script.string:
            continue
        try:
            payload = json.loads(script.string)
        except json.JSONDecodeError:
            continue
        for node in _jsonld_objects(payload):
            raw = node.get("inLanguage")
            if isinstance(raw, str):
                lang = normalize_language_code(raw)
                if lang:
                    return lang
            if isinstance(raw, dict):
                lang = normalize_language_code(str(raw.get("@id", "")))
                if lang:
                    return lang
    return ""


def _jsonld_objects(raw):
    if isinstance(raw, list):
        objects = []
        for item in raw:
            objects.extend(_jsonld_objects(item))
        return objects
    if not isinstance(raw, dict):
        return []
    if "@graph" in raw:
        return _jsonld_objects(raw["@graph"])
    return [raw]


def detect_language_from_text(text: str) -> str:
    sample = " ".join(text.split())[:5000]
    if len(sample) < _MIN_TEXT_CHARS:
        return ""
    try:
        candidates = detect_langs(sample)
    except LangDetectException:
        return ""
    if not candidates:
        return ""
    best = candidates[0]
    if best.prob < _MIN_DETECT_CONFIDENCE:
        return ""
    return normalize_language_code(best.lang)


def extract_language(html: str, body_text: str = "") -> str:
    for detector in (
        lambda: _language_from_html_tag(html),
        lambda: _language_from_head(BeautifulSoup(html, "lxml", parse_only=_HEAD)),
        lambda: _language_from_jsonld(BeautifulSoup(html, "lxml", parse_only=_HEAD)),
        lambda: detect_language_from_text(body_text),
    ):
        language = detector()
        if language:
            return language
    return ""


_CRAWL_LANGUAGES = frozenset({"en"})


def is_crawlable_language(language: str) -> bool:
    """Allow English pages and pages with unknown language."""
    if not language:
        return True
    return language in _CRAWL_LANGUAGES


def detect_page_language(content: str, content_type: str | None = None) -> str:
    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime in {
            "text/plain",
            "text/markdown",
            "text/x-markdown",
            "application/markdown",
        }:
            return detect_language_from_text(content)
    return extract_language(content)
