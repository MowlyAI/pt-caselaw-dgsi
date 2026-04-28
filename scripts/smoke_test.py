"""Smoke test: scrape 1 listing page worth of STJ docs in parallel and report."""
import asyncio
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scraper.config import BASE_URL, DATABASES
from scraper.scraper import decode_content, fetch_with_retry, get_doc_links_from_page, parse_document


async def main():
    db = next(d for d in DATABASES if d["short"] == "STJ")
    start_url = f"{BASE_URL}/{db['db']}/{db['view_id']}?OpenView&Start=1"

    connector = aiohttp.TCPConnector(limit=25, ssl=False)
    async with aiohttp.ClientSession(
        connector=connector,
        headers={"User-Agent": "Mozilla/5.0 (DGSI-Research-Bot/1.0)"},
        timeout=aiohttp.ClientTimeout(total=60),
    ) as session:
        doc_urls, next_url = await get_doc_links_from_page(session, start_url, db)
        print(f"listing: {len(doc_urls)} doc urls, next={bool(next_url)}")
        if not doc_urls:
            return
        sample = doc_urls[:10]

        async def fetch_one(url):
            content = await fetch_with_retry(session, url)
            if not content:
                return url, None
            return url, parse_document(decode_content(content), url, db)

        results = await asyncio.gather(*[fetch_one(u) for u in sample])

    for url, doc in results:
        if not doc:
            print(f"FAIL {url}")
            continue
        ft = doc["full_text"]
        has_proc = "Processo" in ft
        has_rel = "Relator" in ft
        has_dec = "Decisão Texto Integral" in ft or "Acordam" in ft or "RELATÓRIO" in ft.upper()
        print(f"OK {doc['doc_id']:40s} len={len(ft):>6}  "
              f"Processo={'Y' if has_proc else 'N'}  "
              f"Relator={'Y' if has_rel else 'N'}  "
              f"DecTxt={'Y' if has_dec else 'N'}")

    sample_doc = next((d for _, d in results if d), None)
    if sample_doc:
        print("\n--- first 1200 chars of full_text of first doc ---")
        print(sample_doc["full_text"][:1200])


if __name__ == "__main__":
    asyncio.run(main())
