"""BFS crawler: given seed URLs, crawls to configurable depth."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urlparse

from research_pack.crawl.extract import (
    extract_html,
    extract_pdf_text,
    is_same_domain,
    looks_like_html,
    looks_like_pdf,
)
from research_pack.crawl.fetcher import Fetcher, FetchResult
from research_pack.crawl.robots import is_allowed
from research_pack.utils.logging import get_logger

logger = get_logger("crawl.crawler")


@dataclass
class CrawledPage:
    url: str
    final_url: str
    title: str
    text: str
    html: Optional[str]
    content_type: str
    pdf_text_unavailable: bool = False
    error: Optional[str] = None
    depth: int = 0


ProgressCallback = Callable[[str, int, int], None]


async def crawl(
    seed_urls: list[str],
    depth: int = 1,
    max_pages: int = 40,
    max_per_domain: int = 2,
    rate_delay: float = 1.0,
    timeout: float = 20.0,
    keep_html: bool = True,
    progress: Optional[ProgressCallback] = None,
) -> list[CrawledPage]:
    """
    BFS crawl starting from seed_urls.

    Args:
        seed_urls:      Starting URLs (depth 0).
        depth:          How many BFS levels to follow links (0 = seeds only, 1 = seeds + their links).
        max_pages:      Hard cap on total pages fetched.
        max_per_domain: Max concurrent connections per domain.
        rate_delay:     Min seconds between requests to same domain.
        timeout:        HTTP read timeout per request.
        keep_html:      Whether to store raw HTML.
        progress:       Optional callback(message, current, total).
    """
    pages: list[CrawledPage] = []
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(u, 0) for u in seed_urls]

    async with Fetcher(
        max_per_domain=max_per_domain,
        rate_delay=rate_delay,
        timeout=timeout,
    ) as fetcher:

        while queue and len(pages) < max_pages:
            # Process current BFS level in parallel
            current_level = queue[:]
            queue = []

            # Filter already-visited
            to_fetch = [(u, d) for u, d in current_level if u not in visited]
            if not to_fetch:
                break

            if progress:
                progress(f"Crawling {len(to_fetch)} URLs at depth {to_fetch[0][1]}", len(pages), max_pages)

            tasks = [
                _fetch_one(url, dep, fetcher, keep_html)
                for url, dep in to_fetch
                if len(pages) + len(to_fetch) <= max_pages
            ]

            results: list[tuple[CrawledPage, list[str]]] = await asyncio.gather(*tasks, return_exceptions=False)

            for page, outlinks in results:
                visited.add(page.url)
                if page.final_url != page.url:
                    visited.add(page.final_url)
                pages.append(page)

                # Enqueue same-domain links if depth allows
                if page.depth < depth and not page.error:
                    for link in outlinks:
                        if link not in visited and is_same_domain(page.url, link):
                            queue.append((link, page.depth + 1))

                if len(pages) >= max_pages:
                    break

    logger.info("Crawl complete: %d pages fetched, %d unique URLs visited", len(pages), len(visited))
    return pages


async def _fetch_one(
    url: str,
    dep: int,
    fetcher: Fetcher,
    keep_html: bool,
) -> tuple[CrawledPage, list[str]]:
    """Fetch a single URL and return (CrawledPage, outlinks)."""
    # Best-effort robots check
    allowed = await is_allowed(url, fetcher.client)
    if not allowed:
        logger.debug("robots.txt disallows: %s", url)
        return CrawledPage(
            url=url, final_url=url, title="", text="",
            html=None, content_type="",
            error="disallowed by robots.txt", depth=dep,
        ), []

    result = await fetcher.fetch(url)

    if result.error or result.status_code == 0:
        return CrawledPage(
            url=url, final_url=result.final_url,
            title="", text="", html=None, content_type="",
            error=result.error or f"HTTP {result.status_code}", depth=dep,
        ), []

    if result.status_code >= 400:
        return CrawledPage(
            url=url, final_url=result.final_url,
            title="", text="", html=None, content_type=result.content_type,
            error=f"HTTP {result.status_code}", depth=dep,
        ), []

    ct = result.content_type

    # ---- HTML ----
    if looks_like_html(ct) or (not looks_like_pdf(ct, url) and ct == ""):
        try:
            html_str = result.content.decode("utf-8", errors="replace")
        except Exception:
            html_str = ""
        title, text, outlinks = extract_html(html_str, base_url=result.final_url)
        return CrawledPage(
            url=url,
            final_url=result.final_url,
            title=title,
            text=text,
            html=html_str if keep_html else None,
            content_type="text/html",
            depth=dep,
        ), outlinks

    # ---- PDF ----
    if looks_like_pdf(ct, url):
        text = extract_pdf_text(result.content)
        if text:
            return CrawledPage(
                url=url, final_url=result.final_url,
                title="", text=text, html=None,
                content_type="application/pdf", depth=dep,
            ), []
        else:
            return CrawledPage(
                url=url, final_url=result.final_url,
                title="", text="", html=None,
                content_type="application/pdf",
                pdf_text_unavailable=True, depth=dep,
            ), []

    # ---- Unknown content type ----
    logger.debug("Skipping unsupported content type %s for %s", ct, url)
    return CrawledPage(
        url=url, final_url=result.final_url,
        title="", text="", html=None, content_type=ct,
        error=f"unsupported content-type: {ct}", depth=dep,
    ), []
