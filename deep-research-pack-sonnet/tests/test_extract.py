"""Tests for HTML/PDF content extraction."""

import pytest
from research_pack.crawl.extract import (
    extract_html,
    _normalize_whitespace,
    is_same_domain,
    looks_like_html,
    looks_like_pdf,
)


def test_normalize_whitespace():
    text = "hello   world\n\n\n\nextra lines\t\there"
    result = _normalize_whitespace(text)
    assert "   " not in result
    assert "\n\n\n" not in result
    assert "\t\t" not in result


def test_extract_html_basic():
    html = """
    <html>
    <head><title>Test Page</title></head>
    <body>
      <nav>Navigation menu</nav>
      <main>
        <h1>Main Content</h1>
        <p>This is the main article content with useful information.</p>
      </main>
      <footer>Footer text</footer>
      <script>var x = 1;</script>
    </body>
    </html>
    """
    title, text, outlinks = extract_html(html, "https://example.com")
    assert "Main Content" in text or "main article content" in text
    assert "var x = 1" not in text


def test_extract_html_outlinks():
    html = """
    <html><body>
      <a href="https://external.com/page">External</a>
      <a href="/local/path">Local</a>
      <a href="https://example.com/other">Same domain</a>
    </body></html>
    """
    _, _, outlinks = extract_html(html, "https://example.com")
    assert "https://external.com/page" in outlinks
    assert "https://example.com/local/path" in outlinks
    assert "https://example.com/other" in outlinks


def test_is_same_domain():
    assert is_same_domain("https://example.com/a", "https://example.com/b")
    assert not is_same_domain("https://example.com", "https://other.com")


def test_looks_like_html():
    assert looks_like_html("text/html; charset=utf-8")
    assert looks_like_html("application/xhtml+xml")
    assert not looks_like_html("application/pdf")


def test_looks_like_pdf():
    assert looks_like_pdf("application/pdf", "https://example.com/doc")
    assert looks_like_pdf("application/octet-stream", "https://example.com/file.pdf")
    assert not looks_like_pdf("text/html", "https://example.com/page")
