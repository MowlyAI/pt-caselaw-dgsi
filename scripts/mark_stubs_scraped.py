"""Enumerate remaining missing docs, verify they're empty stubs, and mark them in state.

Any DGSI URL that returns <200 chars of body text is published by DGSI as a
metadata-only stub (no sumário, no decision body). Record them in state so the
scraper doesn't keep flagging them as missing.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scraper.config import DATABASES
from scraper.scraper import decode_content, extract_full_text, fetch_with_retry
from scripts.rescrape_missing import enumerate_listing
from bs4 import BeautifulSoup

STATE_FILE = Path("data/scraper_state.json")


async def process_db(session, db):
    urls = await enumerate_listing(session, db)
    state = json.load(open(STATE_FILE))
    have = set(state["scraped_doc_ids"].get(db["db"], []))
    missing = [u for u in urls if u.rsplit("/", 1)[-1].split("?")[0].lower() not in have]
    if not missing:
        print(f"{db['short']}: nothing to mark")
        return state
    stub_ids, content_ids = [], []
    for u in missing:
        raw = await fetch_with_retry(session, u)
        doc_id = u.rsplit("/", 1)[-1].split("?")[0].lower()
        if not raw:
            print(f"  [NO RESPONSE] {doc_id}")
            continue
        body_text = extract_full_text(BeautifulSoup(decode_content(raw), "lxml"))
        if len(body_text) < 200:
            stub_ids.append(doc_id)
        else:
            content_ids.append(doc_id)
    print(f"{db['short']}: {len(stub_ids)} empty stubs, {len(content_ids)} with content")
    if content_ids:
        print("  WITH CONTENT (not a stub):")
        for cid in content_ids:
            print(f"    {cid}")
    state["scraped_doc_ids"].setdefault(db["db"], [])
    state["scraped_doc_ids"][db["db"]] = list(set(state["scraped_doc_ids"][db["db"]]) | set(stub_ids))
    state.setdefault("empty_stub_ids", {}).setdefault(db["db"], [])
    state["empty_stub_ids"][db["db"]] = list(set(state["empty_stub_ids"][db["db"]]) | set(stub_ids))
    return state


async def main():
    connector = aiohttp.TCPConnector(limit=20, ssl=False)
    headers = {"User-Agent": "Mozilla/5.0 (DGSI-Rescue-Bot/1.0)"}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        for db in DATABASES:
            if db["short"] != "TCAS":
                continue
            state = await process_db(session, db)
            tmp = STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, separators=(",", ":")))
            tmp.replace(STATE_FILE)
            print(f"state updated: {len(state['scraped_doc_ids'][db['db']]):,} total, "
                  f"{len(state.get('empty_stub_ids', {}).get(db['db'], [])):,} stubs")


if __name__ == "__main__":
    asyncio.run(main())
