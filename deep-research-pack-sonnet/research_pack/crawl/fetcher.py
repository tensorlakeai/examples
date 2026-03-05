"""Async HTTP fetcher with rate limiting, per-domain concurrency, retries."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import httpx
import tldextract

from research_pack.utils.logging import get_logger

logger = get_logger("crawl.fetcher")

_USER_AGENT = (
    "Mozilla/5.0 (compatible; ResearchPackBot/1.0; +https://github.com/tensorlake)"
)

_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=5.0)
_DEFAULT_RETRIES = 2
_DEFAULT_BACKOFF = 1.5  # seconds, exponential base


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int
    content_type: str
    content: bytes
    error: Optional[str] = None
    elapsed: float = 0.0


@dataclass
class _DomainState:
    sem: asyncio.Semaphore
    last_request: float = 0.0
    rate_delay: float = 1.0  # seconds between requests to same domain


class Fetcher:
    """
    Async HTTP fetcher with:
    - per-domain Semaphore (max_per_domain concurrent connections)
    - polite rate limit (min delay between requests to same domain)
    - configurable timeout and retries
    """

    def __init__(
        self,
        max_per_domain: int = 2,
        rate_delay: float = 1.0,
        timeout: float = 20.0,
        max_retries: int = _DEFAULT_RETRIES,
    ):
        self._max_per_domain = max_per_domain
        self._rate_delay = rate_delay
        self._timeout = httpx.Timeout(connect=10.0, read=timeout, write=10.0, pool=5.0)
        self._max_retries = max_retries
        self._domain_states: dict[str, _DomainState] = defaultdict(
            lambda: _DomainState(
                sem=asyncio.Semaphore(self._max_per_domain),
                rate_delay=self._rate_delay,
            )
        )
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "Fetcher":
        self._client = httpx.AsyncClient(
            headers={"User-Agent": _USER_AGENT},
            timeout=self._timeout,
            follow_redirects=True,
            max_redirects=5,
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()

    def _domain_key(self, url: str) -> str:
        ext = tldextract.extract(url)
        return ext.registered_domain or urlparse(url).netloc

    async def fetch(self, url: str) -> FetchResult:
        key = self._domain_key(url)
        state = self._domain_states[key]

        async with state.sem:
            # Polite delay
            wait = state.rate_delay - (time.monotonic() - state.last_request)
            if wait > 0:
                await asyncio.sleep(wait)

            for attempt in range(self._max_retries + 1):
                try:
                    t0 = time.monotonic()
                    resp = await self._client.get(url)
                    state.last_request = time.monotonic()
                    content_type = resp.headers.get("content-type", "")
                    return FetchResult(
                        url=url,
                        final_url=str(resp.url),
                        status_code=resp.status_code,
                        content_type=content_type,
                        content=resp.content,
                        elapsed=time.monotonic() - t0,
                    )
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt == self._max_retries:
                        logger.warning("Fetch failed %s after %d retries: %s", url, attempt, exc)
                        return FetchResult(
                            url=url,
                            final_url=url,
                            status_code=0,
                            content_type="",
                            content=b"",
                            error=str(exc),
                        )
                    backoff = _DEFAULT_BACKOFF ** (attempt + 1)
                    logger.debug("Retry %d for %s in %.1fs", attempt + 1, url, backoff)
                    await asyncio.sleep(backoff)
                except Exception as exc:
                    return FetchResult(
                        url=url,
                        final_url=url,
                        status_code=0,
                        content_type="",
                        content=b"",
                        error=str(exc),
                    )

        # unreachable
        raise RuntimeError("fetch loop exhausted")

    @property
    def client(self) -> httpx.AsyncClient:
        assert self._client is not None, "Fetcher must be used as async context manager"
        return self._client
