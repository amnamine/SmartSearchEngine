"""Download GSMArena brand listings and save gsmarena_devices.json (slug + fuzzy index)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

from gsmarena_crawl import build_full_index, save_index

DEFAULT_OUT = Path("gsmarena_devices.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--delay", type=float, default=0.35)
    ap.add_argument("--brand", action="append", help="Only crawl given brand key(s), e.g. samsung")
    args = ap.parse_args()

    filt = set(b.lower() for b in args.brand) if args.brand else None
    session = requests.Session()
    try:
        slug, by_br, pages = build_full_index(session, delay=args.delay, brand_filter=filt)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
    save_index(args.out, slug, by_br, pages)
    print(f"[OK] {len(slug)} slugs, {len(by_br)} brands -> {args.out}")


if __name__ == "__main__":
    main()
