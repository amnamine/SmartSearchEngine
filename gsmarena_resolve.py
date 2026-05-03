"""Helpers to turn GSMArena image / pictures URLs into canonical spec page URLs."""
from __future__ import annotations

import json
import re
from pathlib import Path

PHONE_PAGE_RE = re.compile(
    r"https?://(?:www\.)?gsmarena\.com/([a-z0-9_]+)-(\d+)\.php",
    re.IGNORECASE,
)


def pictures_url_to_spec_url(url: str) -> str | None:
    if not url or "gsmarena.com" not in url.lower():
        return None
    m = re.search(
        r"(https?://(?:www\.)?gsmarena\.com/)([a-z0-9_]+)-pictures-(\d+\.php)",
        url,
        re.I,
    )
    if m:
        return f"{m.group(1)}{m.group(2)}-{m.group(3)}"
    m = re.search(
        r"(https?://(?:www\.)?gsmarena\.com/)([a-z0-9_]+)-pictures-(\d+)\.php",
        url,
        re.I,
    )
    if m:
        return f"{m.group(1)}{m.group(2)}-{m.group(3)}"
    m = PHONE_PAGE_RE.match(url.split("?")[0].strip())
    if m and "pictures" not in url.lower() and "price" not in url.lower() and "reviews" not in url.lower():
        return url.split("?")[0].strip()
    return None


def load_link_cache(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items() if v}
    except (json.JSONDecodeError, OSError):
        return {}
