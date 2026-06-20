from semsearch.crawl.main_content import extract_main_text
from semsearch.crawl.metadata import extract_page_metadata


def test_extract_main_text_prefers_main_over_nav():
    html = """
    <html><body>
      <header>
        <nav>
          <button><span class="sr-only">Open menu</span></button>
          <a href="/about">About</a>
          <a href="/apply">Apply</a>
          <a href="/faq">FAQ</a>
        </nav>
      </header>
      <main>
        <h1>Apply to Y Combinator</h1>
        <p>To apply for the Y Combinator program, submit an application form.</p>
        <p>We accept companies in batches four times a year.</p>
      </main>
      <footer>© 2026 Y Combinator</footer>
    </body></html>
    """
    text = extract_main_text(html)

    assert "Apply to Y Combinator" in text
    assert "submit an application form" in text
    assert "Open menu" not in text
    assert "What Happens at YC" not in text


def test_extract_main_text_ignores_nav_links_in_body():
    html = """
    <html><body>
      <nav><a href="/about">About</a><a href="/apply">Apply</a></nav>
      <main><p>Main article content without navigation chrome.</p></main>
    </body></html>
    """
    meta = extract_page_metadata(html, "https://www.ycombinator.com/apply")

    assert "Main article content" in meta.body_excerpt
    assert "Open menu" not in meta.body_excerpt


def test_extract_main_text_falls_back_to_largest_text_block():
    html = """
    <html><body>
      <div class="navbar"><a href="/a">One</a><a href="/b">Two</a><a href="/c">Three</a></div>
      <div>
        <p>This is the primary page content with enough text to be selected as the main block.</p>
        <p>It describes the topic in multiple sentences for the extractor to prefer it.</p>
      </div>
    </body></html>
    """
    text = extract_main_text(html)

    assert "primary page content" in text
    assert "One Two Three" not in text
