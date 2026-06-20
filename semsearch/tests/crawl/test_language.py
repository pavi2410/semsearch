from semsearch.crawl.language import (
    detect_language_from_text,
    detect_page_language,
    extract_language,
    is_crawlable_language,
    normalize_language_code,
)


def test_normalize_language_code():
    assert normalize_language_code("en-US") == "en"
    assert normalize_language_code("zh_CN") == "zh"
    assert normalize_language_code("EN") == "en"
    assert normalize_language_code("eng") == ""


def test_extract_language_from_html_lang():
    html = '<html lang="en"><head></head><body><p>Hello world</p></body></html>'
    assert extract_language(html, "Hello world") == "en"


def test_extract_language_from_meta_tags():
    html = """
    <html>
      <head>
        <meta property="og:locale" content="fr_FR">
      </head>
      <body><p>Bonjour le monde</p></body>
    </html>
    """
    assert extract_language(html, "Bonjour le monde") == "fr"


def test_detect_language_from_text_fallback():
    html = "<html><head></head><body></body></html>"
    french = (
        "Le gouvernement français a annoncé une nouvelle politique économique "
        "pour les petites entreprises dans plusieurs régions du pays."
    )
    assert extract_language(html, french) == "fr"


def test_is_crawlable_language():
    assert is_crawlable_language("") is True
    assert is_crawlable_language("en") is True
    assert is_crawlable_language("fr") is False


def test_detect_page_language_for_plain_text():
    text = (
        "This is a long enough English paragraph to pass language detection "
        "with high confidence for crawl filtering purposes."
    )
    assert detect_page_language(text, "text/plain") == "en"
