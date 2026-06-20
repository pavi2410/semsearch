from semsearch.crawl.metadata import extract_page_metadata
from semsearch.search.snippet import make_snippet


def test_extract_meta_and_og_tags():
    html = """
    <html>
      <head>
        <title>HTML Title</title>
        <meta name="description" content="Meta description">
        <meta property="og:title" content="OG Title">
        <meta property="og:description" content="OG Description">
        <meta property="article:published_time" content="2024-05-01T12:00:00Z">
        <meta property="article:modified_time" content="2024-05-02T08:30:00Z">
        <link rel="canonical" href="https://example.com/article">
      </head>
      <body><p>Body content here.</p></body>
    </html>
    """
    meta = extract_page_metadata(html, "https://example.com/article")

    assert meta.title == "OG Title"
    assert meta.description == "OG Description"
    assert meta.canonical_url == "https://example.com/article"
    assert meta.published_at == "2024-05-01T12:00:00Z"
    assert meta.modified_at == "2024-05-02T08:30:00Z"
    assert "Body content here." in meta.body_text


def test_extract_jsonld_article():
    html = """
    <html>
      <head>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "NewsArticle",
          "headline": "JSON-LD Headline",
          "description": "JSON-LD description",
          "datePublished": "2023-01-15",
          "dateModified": "2023-02-01T10:00:00Z",
          "url": "https://news.example.com/story"
        }
        </script>
      </head>
      <body><p>Story body.</p></body>
    </html>
    """
    meta = extract_page_metadata(html, "https://news.example.com/story")

    assert meta.title == "JSON-LD Headline"
    assert meta.description == "JSON-LD description"
    assert meta.published_at == "2023-01-15T00:00:00Z"
    assert meta.modified_at == "2023-02-01T10:00:00Z"
    assert meta.canonical_url == "https://news.example.com/story"
    assert meta.jsonld_types == ["newsarticle"]


def test_extract_outbound_links_resolves_relative_urls():
    html = """
    <html><body>
      <a href="/docs/page">Relative</a>
      <a href="https://other.example/path">Absolute</a>
      <a href="mailto:test@example.com">Skip</a>
    </body></html>
    """
    meta = extract_page_metadata(html, "https://example.com/root")

    assert meta.outbound_links == [
        "https://example.com/docs/page",
        "https://other.example/path",
    ]


def test_description_falls_back_to_body_excerpt():
    html = "<html><head></head><body><p>Only body text is available here.</p></body></html>"
    meta = extract_page_metadata(html, "https://example.com/page")

    assert meta.description.startswith("Only body text is available")
    assert meta.body_excerpt.startswith("Only body text is available")


def test_make_snippet_prefers_query_match():
    text = "Intro paragraph. Semantic search improves recall for related concepts."
    snippet = make_snippet(text, "semantic search", max_len=40)

    assert "Semantic search" in snippet or "semantic search" in snippet.lower()
