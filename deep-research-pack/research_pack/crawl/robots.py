"""Best-effort robots.txt checking."""

from __future__ import annotations

from urllib.parse import urljoin, urlparse

import httpx

_robots_cache: dict[str, str | None] = {}


def can_fetch(url: str, user_agent: str = "*", timeout: float = 5.0) -> bool:
    """Return *True* unless robots.txt explicitly disallows *url*.

    This is a best-effort, simplified parser — it only looks at
    ``Disallow`` directives for the matching ``User-agent`` block and
    the wildcard block.
    """
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = urljoin(base, "/robots.txt")

    if robots_url not in _robots_cache:
        try:
            resp = httpx.get(robots_url, timeout=timeout, follow_redirects=True)
            _robots_cache[robots_url] = resp.text if resp.status_code == 200 else None
        except Exception:
            _robots_cache[robots_url] = None

    robots_text = _robots_cache[robots_url]
    if robots_text is None:
        return True  # no robots.txt → allow everything

    path = parsed.path or "/"
    current_agents: set[str] = set()
    for line in robots_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "user-agent":
            current_agents = {value}
        elif key == "disallow" and value:
            if user_agent in current_agents or "*" in current_agents:
                if path.startswith(value):
                    return False
    return True
