"""Debug why the 28 TCAS missing docs fetch but produce 0 parsed results."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scraper.config import DATABASES
from scraper.scraper import decode_content, fetch_with_retry, parse_document
from scripts.rescrape_missing import enumerate_listing


async def main():
    db = next(d for d in DATABASES if d["short"] == "TCAS")
    state = json.load(open("data/scraper_state.json"))
    have = set(state["scraped_doc_ids"][db["db"]])
    connector = aiohttp.TCPConnector(limit=20, ssl=False)
    headers = {"User-Agent": "Mozilla/5.0 (DGSI-Rescue-Bot/1.0)"}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        urls = await enumerate_listing(session, db)
        missing = [u for u in urls if u.rsplit("/", 1)[-1].split("?")[0].lower() not in have]
        print(f"missing count: {len(missing)}")
        print("sample missing urls:")
        for u in missing[:5]:
            print(" ", u)
        if not missing:
            return
        ok, fail = 0, 0
        for u in missing:
            raw = await fetch_with_retry(session, u)
            if not raw:
                print(f"  [NO RESPONSE] {u}")
                fail += 1
                continue
            text = decode_content(raw)
            doc = parse_document(text, u, db)
            if doc:
                ok += 1
                print(f"  [OK len={len(doc['full_text'])}] {u}")
            else:
                fail += 1
                print(f"  [PARSE-NONE rawlen={len(text)}] {u}")
        print(f"\nsummary: ok={ok} fail={fail}")


if __name__ == "__main__":
    asyncio.run(main())
