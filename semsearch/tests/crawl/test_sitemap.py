import httpx
import pytest

from semsearch.crawl.sitemap import SitemapLoader

URLSET = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/a</loc></url>
  <url><loc>https://example.com/b</loc></url>
  <url><loc>https://example.com/c</loc></url>
</urlset>"""

SITEMAPINDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/sitemap1.xml</loc></sitemap>
  <sitemap><loc>https://example.com/sitemap2.xml</loc></sitemap>
</sitemapindex>"""

URLSET_1 = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/page1</loc></url>
</urlset>"""

URLSET_2 = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/page2</loc></url>
</urlset>"""


def make_client(responses: dict[str, tuple[int, str]]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        status, body = responses.get(url, (404, ""))
        return httpx.Response(status, text=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------
# basic urlset parsing
# ------------------------------------------------------------------


async def test_urlset_returns_all_locs():
    client = make_client({"https://example.com/sitemap.xml": (200, URLSET)})
    loader = SitemapLoader(client)
    urls = await loader.load("example.com", ["https://example.com/sitemap.xml"])
    assert urls == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]


async def test_fallback_to_default_sitemap_url():
    client = make_client({"https://example.com/sitemap.xml": (200, URLSET)})
    loader = SitemapLoader(client)
    urls = await loader.load("example.com", [])  # no sitemap URLs provided
    assert len(urls) == 3


# ------------------------------------------------------------------
# sitemapindex recursion
# ------------------------------------------------------------------


async def test_sitemapindex_recurses_into_children():
    client = make_client(
        {
            "https://example.com/sitemap_index.xml": (200, SITEMAPINDEX),
            "https://example.com/sitemap1.xml": (200, URLSET_1),
            "https://example.com/sitemap2.xml": (200, URLSET_2),
        }
    )
    loader = SitemapLoader(client)
    urls = await loader.load("example.com", ["https://example.com/sitemap_index.xml"])
    assert "https://example.com/page1" in urls
    assert "https://example.com/page2" in urls


async def test_sitemapindex_deduplicates_child_sitemaps():
    # same child listed twice
    index = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/sitemap1.xml</loc></sitemap>
  <sitemap><loc>https://example.com/sitemap1.xml</loc></sitemap>
</sitemapindex>"""
    call_count = 0

    def handler(request):
        nonlocal call_count
        url = str(request.url)
        if "sitemap_index" in url:
            return httpx.Response(200, text=index)
        call_count += 1
        return httpx.Response(200, text=URLSET_1)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    loader = SitemapLoader(client)
    await loader.load("example.com", ["https://example.com/sitemap_index.xml"])
    assert call_count == 1


# ------------------------------------------------------------------
# error handling
# ------------------------------------------------------------------


async def test_non_200_response_returns_empty():
    client = make_client({"https://example.com/sitemap.xml": (404, "")})
    loader = SitemapLoader(client)
    urls = await loader.load("example.com", ["https://example.com/sitemap.xml"])
    assert urls == []


async def test_invalid_xml_returns_empty():
    client = make_client({"https://example.com/sitemap.xml": (200, "not xml {{}")})
    loader = SitemapLoader(client)
    urls = await loader.load("example.com", ["https://example.com/sitemap.xml"])
    assert urls == []


async def test_network_error_returns_empty():
    def handler(request):
        raise httpx.ConnectError("refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    loader = SitemapLoader(client)
    urls = await loader.load("example.com", ["https://example.com/sitemap.xml"])
    assert urls == []


async def test_filters_non_http_locs():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/valid</loc></url>
  <url><loc>ftp://example.com/invalid</loc></url>
  <url><loc>mailto:someone@example.com</loc></url>
</urlset>"""
    client = make_client({"https://example.com/sitemap.xml": (200, xml)})
    loader = SitemapLoader(client)
    urls = await loader.load("example.com", ["https://example.com/sitemap.xml"])
    assert urls == ["https://example.com/valid"]
