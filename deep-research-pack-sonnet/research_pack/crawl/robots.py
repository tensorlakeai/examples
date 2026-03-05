"""Best-effort robots.txt checker using protego."""

from __future__ import annotations

import httpx
from urllib.parse import urlparse
from research_pack.utils.logging import get_logger

logger = get_logger("crawl.robots")

_CACHE: dict[str, object] = {}
USER_AGENT = "ResearchPackBot/1.0"


def _base(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


async def is_allowed(url: str, client: httpx.AsyncClient) -> bool:
    """Return True if the URL is allowed by robots.txt (best-effort)."""
    try:
        import protego  # optional dependency
    except ImportError:
        return True  # no protego → skip check

    base = _base(url)
    if base not in _CACHE:
        try:
            resp = await client.get(f"{base}/robots.txt", timeout=5.0)
            _CACHE[base] = protego.Protego.parse(resp.text)
        except Exception as exc:
            logger.debug("robots.txt fetch failed for %s: %s", base, exc)
            _CACHE[base] = None

    rp = _CACHE[base]
    if rp is None:
        return True
    return rp.can_fetch(url, USER_AGENT)
