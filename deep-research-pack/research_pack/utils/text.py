"""Text normalisation and URL helpers."""

from __future__ import annotations

import re
from urllib.parse import urlparse


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip common web boilerplate."""
    text = re.sub(r"\s+", " ", text).strip()
    boilerplate_patterns = [
        r"Cookie\s+(?:Policy|Notice|Consent).*?(?:\.|$)",
        r"Accept\s+(?:All\s+)?Cookies.*?(?:\.|$)",
        r"Subscribe\s+to\s+our\s+newsletter.*?(?:\.|$)",
        r"\u00a9\s*\d{4}.*?(?:\.|$)",
        r"All\s+rights\s+reserved.*?(?:\.|$)",
        r"Privacy\s+Policy.*?Terms\s+of\s+(?:Service|Use)",
        r"Sign\s+up\s+for\s+(?:free|our).*?(?:\.|$)",
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def extract_domain(url: str) -> str:
    """Return the network-location part of a URL."""
    parsed = urlparse(url)
    return parsed.netloc or parsed.hostname or url


def truncate(text: str, max_chars: int = 500) -> str:
    """Truncate *text* to *max_chars*, adding an ellipsis when shortened."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\u2026"
