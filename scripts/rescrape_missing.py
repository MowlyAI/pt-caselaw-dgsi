"""
Find and re-scrape documents that the main scraper missed.

1. For each requested DB, enumerate every listing page (Start=1..N, stepping by
   the listing page size) collecting every doc_id linked from any "OpenDocument"
   anchor. Stop when we stop seeing NEW doc_ids (2 empty pages in a row).
2. Diff that set against `data/scraper_state.json.scraped_doc_ids[<db>]`.
3. Fetch the missing documents in parallel with a global semaphore and append
   them to the existing raw shards (next chunk file) for that DB.
4. Update state file atomically.

Usage:
    python scripts/rescrape_missing.py --db STJ --db TCAS --db TCAN
    python scripts/rescrape_missing.py --all-incomplete
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import aiohttp
import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

load_dotenv(".env.local")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scraper.config import BASE_URL, DATABASES
from scraper.scraper import decode_content, fetch_with_retry, parse_document

console = Console()

DATA_DIR = Path("data/raw")
STATE_FILE = Path("data/scraper_state.json")
PAGE_SIZE = 100  # DGSI returns ~100 docs per listing page
_DOC_ID_RE = re.compile(r"/([a-f0-9]{32})\?OpenDocument", re.IGNORECASE)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"scraped_doc_ids": {}, "completed_dbs": []}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, separators=(",", ":")))
    tmp.replace(STATE_FILE)


def next_chunk_path(db_short: str) -> Path:
    db_dir = DATA_DIR / db_short
    db_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(db_dir.glob("chunk_*.jsonl"))
    idx = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
    return db_dir / f"chunk_{idx:04d}.jsonl"


async def enumerate_listing(
    session: aiohttp.ClientSession, db: dict, max_empty_pages: int = 3
) -> list[str]:
    """Walk Start=1..N by PAGE_SIZE and return every unique doc URL seen."""
    seen: set[str] = set()
    urls: list[str] = []
    empty_streak = 0
    start = 1
    pages_visited = 0
    while empty_streak < max_empty_pages:
        listing_url = f"{BASE_URL}/{db['db']}/{db['view_id']}?OpenView&Start={start}"
        raw = await fetch_with_retry(session, listing_url)
        pages_visited += 1
        if not raw:
            empty_streak += 1
            start += PAGE_SIZE
            continue
        text = decode_content(raw)
        page_new = 0
        for m in _DOC_ID_RE.finditer(text):
            doc_id = m.group(1).lower()
            if doc_id not in seen:
                seen.add(doc_id)
                urls.append(f"{BASE_URL}/{db['db']}/{db['view_id']}/{doc_id}?OpenDocument")
                page_new += 1
        if page_new == 0:
            empty_streak += 1
        else:
            empty_streak = 0
        if pages_visited % 50 == 0:
            console.print(f"  {db['short']}: visited {pages_visited} pages, {len(urls)} docs found")
        start += PAGE_SIZE
    console.print(f"[green]{db['short']}: enumeration done — {len(urls)} docs, {pages_visited} pages[/green]")
    return urls


async def fetch_missing(
    session: aiohttp.ClientSession, urls: list[str], db: dict,
    sem: asyncio.Semaphore, progress: Progress, task_id,
) -> list[dict]:
    results: list[dict] = []

    async def _one(u: str):
        async with sem:
            content = await fetch_with_retry(session, u)
            if content:
                doc = parse_document(decode_content(content), u, db)
                if doc:
                    results.append(doc)
            progress.advance(task_id)

    await asyncio.gather(*[_one(u) for u in urls])
    return results


@click.command()
@click.option("--db", "dbs_arg", multiple=True, help="DB short code (can repeat)")
@click.option("--all-incomplete", is_flag=True, help="all DBs where scraped < approx_count*0.99")
@click.option("--concurrency", default=40, type=int)
def main(dbs_arg: tuple, all_incomplete: bool, concurrency: int):
    state = load_state()
    scraped = state.get("scraped_doc_ids", {})

    if all_incomplete:
        dbs = [d for d in DATABASES if len(scraped.get(d["db"], [])) < d["approx_count"] * 0.99]
    else:
        wanted = {s.upper() for s in dbs_arg}
        dbs = [d for d in DATABASES if d["short"].upper() in wanted]

    if not dbs:
        console.print("[yellow]no DBs selected[/yellow]"); sys.exit(0)

    console.print(f"[cyan]Targets:[/cyan] {', '.join(d['short'] for d in dbs)}")

    async def run():
        connector = aiohttp.TCPConnector(limit=concurrency, ssl=False)
        headers = {"User-Agent": "Mozilla/5.0 (DGSI-Rescue-Bot/1.0)"}
        sem = asyncio.Semaphore(concurrency)
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            for db in dbs:
                t0 = time.perf_counter()
                urls = await enumerate_listing(session, db)
                have = set(scraped.get(db["db"], []))
                missing_urls = [u for u in urls if u.rsplit("/", 1)[-1].split("?")[0].lower() not in have]
                console.print(
                    f"[cyan]{db['short']}[/cyan]: listing={len(urls):,}  have={len(have):,}  "
                    f"missing=[bold]{len(missing_urls):,}[/bold]  (enum {time.perf_counter()-t0:.1f}s)"
                )
                if not missing_urls:
                    continue
                with Progress(
                    TextColumn(f"[bold]{db['short']}"), BarColumn(), TaskProgressColumn(),
                    TimeElapsedColumn(), console=console,
                ) as progress:
                    tid = progress.add_task("fetch", total=len(missing_urls))
                    docs = await fetch_missing(session, missing_urls, db, sem, progress, tid)
                out = next_chunk_path(db["short"])
                with open(out, "w", encoding="utf-8") as f:
                    for d in docs:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
                new_ids = [d["doc_id"] for d in docs]
                scraped[db["db"]] = list(set(scraped.get(db["db"], [])) | set(new_ids))
                save_state({"scraped_doc_ids": scraped, "completed_dbs": state.get("completed_dbs", [])})
                console.print(f"[green]{db['short']}: wrote {len(docs):,} docs → {out.name}[/green]")

    asyncio.run(run())


if __name__ == "__main__":
    main()
