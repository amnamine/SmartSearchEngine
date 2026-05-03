"""
Crawl GSMArena brand listing pages (works with a normal browser User-Agent from residential IPs).
Builds a slug -> canonical spec URL index and per-brand device lists for fuzzy matching.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

BASE = "https://www.gsmarena.com/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

PHONE_HREF_RE = re.compile(
    r'<a href="([a-z0-9_]+-\d+\.php)"[^>]*>(?:<img[^>]*>)?<strong><span>([^<]*)</span></strong></a>',
    re.I,
)
NEXT_PAGE_RE = re.compile(
    r'<a href="([^"]+)" class="prevnextbutton" title="Next page">',
    re.I,
)
BRAND_PAGES_RE = re.compile(r'href="([a-z0-9]+-phones-\d+\.php)"', re.I)


def fetch_text(url: str, session: requests.Session, retries: int = 4) -> str:
    last_err: Exception | None = None
    wait = 2.0
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=45)
            r.raise_for_status()
            return r.text
        except (requests.RequestException, OSError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(wait)
                wait = min(wait * 1.5, 30.0)
    assert last_err is not None
    raise last_err


def discover_brand_pages(session: requests.Session) -> dict[str, str]:
    """Map brand key (e.g. samsung) -> first listing path (samsung-phones-9.php)."""
    html = fetch_text(BASE, session)
    paths = sorted(set(BRAND_PAGES_RE.findall(html)))
    out: dict[str, str] = {}
    for p in paths:
        m = re.match(r"([a-z]+)-phones-(\d+)\.php", p, re.I)
        if m:
            out[m.group(1).lower()] = p
    return out


def parse_phones_on_page(html: str) -> list[tuple[str, str, str]]:
    """Returns list of (relative_href, slug_without_id, display_span)."""
    found = []
    for m in PHONE_HREF_RE.finditer(html):
        href, span = m.group(1), m.group(2).strip()
        stem = href.rsplit(".", 1)[0]
        base_slug = re.sub(r"-\d+$", "", stem)
        found.append((href, base_slug, span))
    for m in re.finditer(r'<a href="([a-z0-9_]+-\d+\.php)"', html, re.I):
        href = m.group(1)
        if "-phones-" in href or "compare" in href or "review" in href:
            continue
        stem = href.rsplit(".", 1)[0]
        base_slug = re.sub(r"-\d+$", "", stem)
        if not any(x[0] == href for x in found):
            found.append((href, base_slug, ""))
    return found


def crawl_brand_devices(
    first_path: str,
    session: requests.Session,
    delay: float,
) -> list[dict]:
    devices: list[dict] = []
    current = first_path
    seen_pages: set[str] = set()
    while current and current not in seen_pages:
        seen_pages.add(current)
        url = urljoin(BASE, current)
        html = fetch_text(url, session)
        for href, base_slug, span in parse_phones_on_page(html):
            full = urljoin(BASE, href)
            devices.append(
                {"href": href, "slug": base_slug, "span": span, "url": full}
            )
        m = NEXT_PAGE_RE.search(html)
        if not m:
            break
        nxt = m.group(1)
        if not nxt or nxt == "#":
            break
        if nxt.startswith("http"):
            from urllib.parse import urlparse

            path = urlparse(nxt).path.lstrip("/")
            nxt_path = path or None
        else:
            nxt_path = nxt
        if not nxt_path or nxt_path in seen_pages:
            break
        current = nxt_path
        time.sleep(delay)
    return devices


def build_full_index(
    session: requests.Session,
    delay: float = 0.35,
    brand_filter: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, list[dict]], dict[str, str]]:
    """
    Returns (slug_index, by_brand, brand_pages) where slug_index maps base slug -> spec URL.
    """
    brand_pages = discover_brand_pages(session)
    slug_index: dict[str, str] = {}
    by_brand: dict[str, list[dict]] = {}

    for bkey, first_path in sorted(brand_pages.items()):
        if brand_filter and bkey not in brand_filter:
            continue
        devs = crawl_brand_devices(first_path, session, delay=delay)
        by_brand[bkey] = devs
        for d in devs:
            slug_index[d["slug"].lower()] = d["url"]
        print(f"[gsmarena] {bkey}: {len(devs)} devices", flush=True)
        time.sleep(delay)
    return slug_index, by_brand, brand_pages


def save_index(
    path: Path,
    slug_index: dict[str, str],
    by_brand: dict[str, list[dict]],
    brand_pages: dict[str, str],
) -> None:
    payload = {
        "slug_index": slug_index,
        "by_brand": by_brand,
        "brand_pages": brand_pages,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_index(path: Path) -> tuple[dict[str, str], dict[str, list[dict]], dict[str, str]]:
    if not path.is_file():
        return {}, {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        data.get("slug_index", {}),
        data.get("by_brand", {}),
        data.get("brand_pages", {}),
    )


def bigpic_url_to_base_slug(image_url: str) -> str | None:
    if not image_url or "gsmarena.com" not in image_url.lower():
        return None
    m = re.search(r"/([^/]+)\.(jpg|jpeg|png|webp)(\?|$)", image_url, re.I)
    if not m:
        return None
    stem = m.group(1).lower()
    return stem.replace("-", "_")


def normalize_marque_to_brand_key(marque: str, brand_pages: dict[str, str]) -> str | None:
    if not marque:
        return None
    key = re.sub(r"[^a-z0-9]+", "", marque.lower())
    aliases = {
        "redmi": "xiaomi",
        "poco": "xiaomi",
        "honor": "honor",
        "oneplus": "oneplus",
        "oneplusone": "oneplus",
    }
    key = aliases.get(key, key)
    if key in brand_pages:
        return key
    for bk in brand_pages:
        if bk.startswith(key) or key.startswith(bk):
            return bk
    return None


def token_score(name_a: str, name_b: str) -> int:
    ta = [t for t in re.findall(r"[a-z0-9]+", name_a.lower()) if len(t) > 1]
    tb = set(re.findall(r"[a-z0-9]+", name_b.lower()))
    return sum(1 for t in ta if t in tb)


def resolve_url_for_product(
    product_name: str,
    marque: str,
    image_url: str,
    slug_index: dict[str, str],
    by_brand: dict[str, list[dict]],
    brand_pages: dict[str, str],
) -> str | None:
    """Pick best GSMArena spec URL for a catalogue row."""
    bs = bigpic_url_to_base_slug(image_url or "")
    if bs and bs in slug_index:
        return slug_index[bs]
    if bs:
        best_k, best_u, best_len = "", "", -1
        for key, url in slug_index.items():
            if bs == key or key.startswith(bs + "_") or bs.startswith(key + "_"):
                if len(key) > best_len:
                    best_len = len(key)
                    best_k, best_u = key, url
        if best_u:
            return best_u

    bkey = normalize_marque_to_brand_key(marque or "", brand_pages)
    if not bkey:
        return None
    devices = by_brand.get(bkey) or []
    if not devices:
        return None

    best_url = None
    best_score = 0
    pn = (product_name or "").lower()
    mar = (marque or "").strip()
    for d in devices:
        slug = d.get("slug", "")
        span = d.get("span", "")
        title_guess = f"{mar} {span}".strip()
        sc = max(
            token_score(pn, slug.replace("_", " ")),
            token_score(pn, span),
            token_score(pn, title_guess),
        )
        if sc > best_score:
            best_score = sc
            best_url = d.get("url")

    need = 3 if len(pn) > 25 else 2
    if best_score >= need and best_url:
        return best_url
    return None


def guess_marque_from_product_name(product_name: str) -> str:
    parts = (product_name or "").strip().split()
    return parts[0] if parts else ""
