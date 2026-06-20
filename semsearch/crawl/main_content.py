import html as html_mod
import re

from bs4 import BeautifulSoup, SoupStrainer, Tag

_BODY = SoupStrainer("body")

_BOILERPLATE_TAGS = {"nav", "footer", "aside"}
_BOILERPLATE_ROLES = {"navigation", "banner", "contentinfo", "complementary"}
_BOILERPLATE_CLASS_RE = re.compile(
    r"(^|[-_])(nav|navbar|menu|sidebar|breadcrumb|cookie|modal|footer|header-bar)([-_]|$)",
    re.I,
)
_CONTENT_ROOT_SELECTORS: list[tuple[str, dict[str, str]]] = [
    ("main", {}),
    ("article", {}),
    ("div", {"role": "main"}),
    ("section", {"role": "main"}),
    ("div", {"id": "content"}),
    ("div", {"id": "main"}),
    ("div", {"class": "content"}),
]


def _unescape(text: str) -> str:
    return html_mod.unescape(text.strip()) if text else ""


def _tag_role(tag: Tag) -> str:
    attrs = getattr(tag, "attrs", None) or {}
    return str(attrs.get("role", "")).lower()


def _tag_classes(tag: Tag) -> str:
    attrs = getattr(tag, "attrs", None) or {}
    classes = attrs.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    return " ".join(classes).lower()


def _is_page_level_boilerplate(tag: Tag) -> bool:
    if tag.name in _BOILERPLATE_TAGS:
        return True
    if _tag_role(tag) in _BOILERPLATE_ROLES:
        return True
    if tag.name == "header" and tag.find_parent(["main", "article"]) is None:
        return True
    if _BOILERPLATE_CLASS_RE.search(_tag_classes(tag)):
        return True
    return False


def _strip_boilerplate(root: Tag) -> None:
    for tag in root.find_all(["script", "style", "noscript", "svg"]):
        tag.decompose()
    for tag in list(root.find_all(True)):
        if not isinstance(tag, Tag):
            continue
        if _is_page_level_boilerplate(tag):
            tag.decompose()


def _block_score(tag: Tag) -> int:
    text = tag.get_text(separator=" ", strip=True)
    if len(text) < 80:
        return 0
    link_text = " ".join(a.get_text(separator=" ", strip=True) for a in tag.find_all("a"))
    link_ratio = len(link_text) / max(len(text), 1)
    if link_ratio > 0.55:
        return 0
    return len(text) - int(len(link_text) * 0.4)


def _find_content_root(body: Tag) -> Tag | None:
    for tag_name, attrs in _CONTENT_ROOT_SELECTORS:
        tag = body.find(tag_name, attrs)
        if tag is not None and _block_score(tag) > 0:
            return tag

    candidates = body.find_all(["main", "article", "section", "div"])
    scored = [(tag, _block_score(tag)) for tag in candidates]
    scored = [(tag, score) for tag, score in scored if score > 0]
    if not scored:
        return None
    return max(scored, key=lambda item: item[1])[0]


def extract_main_text(html: str) -> str:
    body = BeautifulSoup(html, "lxml", parse_only=_BODY)
    if body.body is None:
        return ""

    work = BeautifulSoup(str(body.body), "lxml").body
    if work is None:
        return ""

    root = _find_content_root(work) or work
    _strip_boilerplate(root)
    return _unescape(root.get_text(separator=" ", strip=True))
