"""Tests for text utilities."""

from research_pack.utils.text import extract_domain, normalize_text, truncate


def test_normalize_collapses_whitespace():
    assert normalize_text("  hello   world  ") == "hello world"


def test_normalize_strips_boilerplate():
    text = "Main content. All rights reserved. More text."
    result = normalize_text(text)
    assert "All rights reserved" not in result
    assert "Main content" in result


def test_normalize_empty():
    assert normalize_text("") == ""


def test_extract_domain():
    assert extract_domain("https://www.example.com/path?q=1") == "www.example.com"
    assert extract_domain("http://blog.ai/post") == "blog.ai"


def test_truncate_short():
    assert truncate("short", 100) == "short"


def test_truncate_long():
    result = truncate("a" * 600, 500)
    assert len(result) == 501  # 500 + ellipsis char
    assert result.endswith("\u2026")
