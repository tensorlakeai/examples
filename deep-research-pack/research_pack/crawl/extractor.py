"""HTML content extraction — readability-lxml with BeautifulSoup fallback."""

from __future__ import annotations

from bs4 import BeautifulSoup


def extract_content(html: str, url: str = "") -> dict[str, str]:
    """Return ``{title, author, published, text}`` from raw *html*.

    Uses *readability-lxml* when available, otherwise falls back to
    BeautifulSoup heuristics.
    """
    soup = BeautifulSoup(html, "lxml")

    # -- metadata ----------------------------------------------------------
    title = _meta(soup, "og:title") or _tag_text(soup, "title")
    author = _meta(soup, "author", attr="name") or ""
    published = (
        _meta(soup, "article:published_time")
        or _meta(soup, "datePublished", attr="name")
        or ""
    )

    # -- main content ------------------------------------------------------
    text = ""
    try:
        from readability import Document

        doc = Document(html)
        main_html = doc.summary()
        main_soup = BeautifulSoup(main_html, "lxml")
        text = main_soup.get_text(separator="\n", strip=True)
        if not title:
            title = doc.short_title()
    except Exception:
        # fallback: strip boilerplate tags
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        body = soup.find("body") or soup
        text = body.get_text(separator="\n", strip=True)

    # collapse blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)

    return {
        "title": title or "",
        "author": author,
        "published": published,
        "text": text,
    }


def extract_links(html: str, base_url: str) -> list[str]:
    """Return de-duplicated, absolute HTTP(S) links found in *html*."""
    from urllib.parse import urljoin, urlparse

    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.scheme not in ("http", "https"):
            continue
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            clean += f"?{parsed.query}"
        if clean not in seen:
            seen.add(clean)
            links.append(clean)
    return links


# -- helpers ---------------------------------------------------------------


def _meta(soup: BeautifulSoup, key: str, attr: str = "property") -> str:
    tag = soup.find("meta", attrs={attr: key})
    if tag and tag.get("content"):
        return tag["content"]
    return ""


def _tag_text(soup: BeautifulSoup, tag_name: str) -> str:
    tag = soup.find(tag_name)
    return tag.get_text(strip=True) if tag else ""
