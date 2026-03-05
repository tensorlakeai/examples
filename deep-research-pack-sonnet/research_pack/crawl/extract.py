"""Content extraction: HTML → clean text, PDF → text."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from research_pack.utils.logging import get_logger

logger = get_logger("crawl.extract")

# Tags whose content is always boilerplate
_STRIP_TAGS = {
    "script", "style", "noscript", "svg", "canvas",
    "nav", "footer", "header", "aside", "form",
    "button", "iframe", "advertisement",
}

_MIN_CONTENT_LEN = 80  # chars; discard pages shorter than this


def extract_html(html: str, base_url: str = "") -> tuple[str, str, list[str]]:
    """
    Extract (title, main_text, outlinks) from raw HTML.

    Returns:
        title       – page title (may be empty)
        text        – cleaned main-body text
        outlinks    – list of absolute URLs found in the page
    """
    # Try readability first for cleaner article text
    try:
        from readability import Document as ReadabilityDoc  # type: ignore
        doc = ReadabilityDoc(html)
        title = doc.title() or ""
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

    # Strip boilerplate tags
    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = _normalize_whitespace(text)

    # Outlinks
    outlinks: list[str] = []
    if base_url:
        for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith(("http://", "https://")):
                outlinks.append(href)
            elif href.startswith("/") and base_url:
                outlinks.append(urljoin(base_url, href))

    return title, text, outlinks


def extract_pdf_text(pdf_bytes: bytes) -> Optional[str]:
    """Extract text from PDF bytes using pdfminer.six (best-effort)."""
    try:
        from pdfminer.high_level import extract_text_to_fp  # type: ignore
        from pdfminer.layout import LAParams
        import io

        output = io.StringIO()
        extract_text_to_fp(
            io.BytesIO(pdf_bytes),
            output,
            laparams=LAParams(),
            output_type="text",
            codec="utf-8",
        )
        text = output.getvalue()
        return _normalize_whitespace(text) if text.strip() else None
    except Exception as exc:
        logger.warning("PDF extraction failed: %s", exc)
        return None


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive blank lines and leading/trailing whitespace."""
    # Replace tabs with spaces
    text = text.replace("\t", " ")
    # Collapse horizontal whitespace runs
    text = re.sub(r" {2,}", " ", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_same_domain(url_a: str, url_b: str) -> bool:
    return urlparse(url_a).netloc == urlparse(url_b).netloc


def looks_like_html(content_type: str) -> bool:
    return "html" in content_type.lower()


def looks_like_pdf(content_type: str, url: str) -> bool:
    return "pdf" in content_type.lower() or url.lower().endswith(".pdf")
