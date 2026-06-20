from semsearch.crawl.language import (
    detect_language_from_text,
    extract_language,
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
