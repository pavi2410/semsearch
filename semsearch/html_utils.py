import html as html_mod
from urllib.parse import urljoin

from bs4 import BeautifulSoup, SoupStrainer

A_TAG = SoupStrainer("a", href=True)
BODY_TEXT_TAGS = SoupStrainer("body")
TITLE_TAG = SoupStrainer("head", recursive=True)


def extract_links(html: str, base_url: str) -> list[str]:
    links: list[str] = []
    for a in BeautifulSoup(html, "lxml", parse_only=A_TAG):
        href = a.get("href")
        if href and href.startswith("http"):
            links.append(urljoin(base_url, href))
    return links


def extract_title(html: str) -> str:
    title_tag = BeautifulSoup(html, "lxml", parse_only=TITLE_TAG).find("title")
    if title_tag and title_tag.string:
        return html_mod.unescape(title_tag.string.strip())
    return ""


def extract_text(html: str) -> str:
    body = BeautifulSoup(html, "lxml", parse_only=BODY_TEXT_TAGS)
    for tag in body.find_all(["script", "style"]):
        tag.decompose()
    return html_mod.unescape(body.get_text(separator=" ", strip=True))


def extract_metadata(html: str) -> tuple[str, str]:
    return extract_title(html), extract_text(html)
