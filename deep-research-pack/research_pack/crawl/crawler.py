"""Web crawler with rate-limiting, retries, and depth traversal."""

from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

import httpx

from research_pack.crawl.extractor import extract_content, extract_links
from research_pack.crawl.robots import can_fetch
from research_pack.models import CrawlResult

logger = logging.getLogger("research_pack.crawl")

# ---------------------------------------------------------------------------
# Per-domain rate-limiting state
# ---------------------------------------------------------------------------
_domain_last_request: dict[str, float] = {}
RATE_LIMIT_SECONDS = 1.0

_USER_AGENT = "ResearchPack/1.0 (research-bot; +https://github.com/tensorlakeai)"
_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"


def _rate_limit(domain: str) -> None:
    now = time.time()
    last = _domain_last_request.get(domain, 0.0)
    wait = RATE_LIMIT_SECONDS - (now - last)
    if wait > 0:
        time.sleep(wait)
    _domain_last_request[domain] = time.time()


# ---------------------------------------------------------------------------
# Single-page fetch
# ---------------------------------------------------------------------------


def fetch_page(
    url: str,
    timeout: float = 15.0,
    retries: int = 2,
) -> CrawlResult:
    """Fetch *url* with retries, timeouts and rate-limiting.

    Returns a :class:`CrawlResult` regardless of outcome (errors are
    captured in ``CrawlResult.error``).
    """
    domain = urlparse(url).netloc
    _rate_limit(domain)

    # robots.txt (best-effort)
    if not can_fetch(url):
        return CrawlResult(url=url, error="blocked_by_robots_txt")

    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                max_redirects=5,
            ) as client:
                resp = client.get(
                    url,
                    headers={"User-Agent": _USER_AGENT, "Accept": _ACCEPT},
                )

            ct = resp.headers.get("content-type", "")
            canonical = str(resp.url) if str(resp.url) != url else None

            if resp.status_code >= 400:
                return CrawlResult(
                    url=url,
                    status=resp.status_code,
                    error=f"http_{resp.status_code}",
                )

            # --- HTML ---------------------------------------------------------
            if "text/html" in ct or "application/xhtml" in ct:
                html = resp.text
                extracted = extract_content(html, url)
                links = extract_links(html, url)
                return CrawlResult(
                    url=url,
                    canonical_url=canonical,
                    content_type="text/html",
                    status=resp.status_code,
                    html=html,
                    text=extracted["text"],
                    title=extracted["title"],
                    author=extracted["author"],
                    published=extracted["published"],
                    links=links,
                )

            # --- PDF ----------------------------------------------------------
            if "application/pdf" in ct:
                return CrawlResult(
                    url=url,
                    canonical_url=canonical,
                    content_type="application/pdf",
                    status=resp.status_code,
                    pdf_text_unavailable=True,
                )

            # --- other text ---------------------------------------------------
            if "text/" in ct:
                return CrawlResult(
                    url=url,
                    canonical_url=canonical,
                    content_type=ct.split(";")[0].strip(),
                    status=resp.status_code,
                    text=resp.text,
                )

            # --- binary / unknown ---------------------------------------------
            return CrawlResult(
                url=url,
                canonical_url=canonical,
                content_type=ct.split(";")[0].strip() or "unknown",
                status=resp.status_code,
            )

        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(2**attempt)

    return CrawlResult(url=url, error=last_error)


# ---------------------------------------------------------------------------
# Breadth-first seed crawl
# ---------------------------------------------------------------------------


def crawl_seed(
    url: str,
    max_depth: int = 1,
    max_pages_per_domain: int = 5,
) -> list[CrawlResult]:
    """Crawl *url* breadth-first up to *max_depth* hops.

    Respects a per-domain page cap of *max_pages_per_domain*.
    """
    results: list[CrawlResult] = []
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(url, 0)]
    domain_counts: dict[str, int] = {}

    while queue:
        current_url, depth = queue.pop(0)
        if current_url in visited:
            continue

        domain = urlparse(current_url).netloc
        if domain_counts.get(domain, 0) >= max_pages_per_domain:
            continue

        visited.add(current_url)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        logger.info("fetch depth=%d url=%s", depth, current_url)
        result = fetch_page(current_url)
        results.append(result)

        if depth < max_depth and result.links:
            for link in result.links[:20]:
                if link not in visited:
                    queue.append((link, depth + 1))

    return results
