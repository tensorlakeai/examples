"""Tests for HTML content extraction."""

from research_pack.crawl.extractor import extract_content, extract_links


_SAMPLE_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <title>Test Page</title>
  <meta property="og:title" content="OG Title" />
  <meta name="author" content="Jane Doe" />
  <meta property="article:published_time" content="2025-01-15" />
</head>
<body>
  <nav>Navigation</nav>
  <article>
    <p>This is the main content of the article. It contains important information.</p>
    <p>Second paragraph with more details about the topic at hand.</p>
  </article>
  <footer>Footer stuff</footer>
</body>
</html>
"""


def test_extract_title():
    result = extract_content(_SAMPLE_HTML)
    assert result["title"] == "OG Title"


def test_extract_author():
    result = extract_content(_SAMPLE_HTML)
    assert result["author"] == "Jane Doe"


def test_extract_published():
    result = extract_content(_SAMPLE_HTML)
    assert result["published"] == "2025-01-15"


def test_extract_text_non_empty():
    result = extract_content(_SAMPLE_HTML)
    assert len(result["text"]) > 20
    assert "main content" in result["text"].lower()


_LINK_HTML = """\
<html><body>
<a href="/page1">Page 1</a>
<a href="https://other.com/x">Other</a>
<a href="mailto:a@b.com">Email</a>
<a href="#fragment">Frag</a>
</body></html>
"""


def test_extract_links():
    links = extract_links(_LINK_HTML, "https://example.com/base")
    # /page1 should resolve to https://example.com/page1
    assert "https://example.com/page1" in links
    assert "https://other.com/x" in links
    # mailto and fragment-only should be excluded
    assert not any("mailto" in l for l in links)
