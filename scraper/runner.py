"""
DGSI scraper orchestrator.

Runs ALL databases in parallel inside a single aiohttp session, sharing one
global semaphore that caps total concurrent HTTP connections. Each DB walks
its own listing pagination and fans out document fetches.

CLI:
    python -m scraper.runner                # all DBs
    python -m scraper.runner --db STJ       # single DB
    python -m scraper.runner --concurrency 80
    python -m scraper.runner --reset
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Optional

import aiohttp
import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from scraper.config import BASE_URL, DATABASES, DOCS_PER_CHUNK
from scraper.scraper import (
    DATA_DIR, STATE_FILE, decode_content, fetch_with_retry,
    get_doc_links_from_page, parse_document,
)

console = Console()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DGSI-Research-Bot/1.0)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8",
}

_state_lock = threading.Lock()


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed_dbs": [], "scraped_doc_ids": {}, "db_progress": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    tmp.replace(STATE_FILE)


class ChunkWriter:
    """Appends JSONL docs to data/raw/<COURT>/chunk_XXXX.jsonl; rolls at N docs."""

    def __init__(self, db_short: str):
        self.db_short = db_short
        self.chunk = 0
        self.count_in_chunk = 0
        self._file = None
        existing = sorted((DATA_DIR / db_short).glob("chunk_*.jsonl")) \
            if (DATA_DIR / db_short).exists() else []
        if existing:
            self.chunk = len(existing) - 1
            with open(existing[-1], "rb") as f:
                self.count_in_chunk = sum(1 for _ in f)
        self._open()

    def _open(self):
        if self._file:
            self._file.close()
        path = DATA_DIR / self.db_short / f"chunk_{self.chunk:04d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    def write(self, doc: dict):
        self._file.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self._file.flush()
        self.count_in_chunk += 1
        if self.count_in_chunk >= DOCS_PER_CHUNK:
            self.chunk += 1
            self.count_in_chunk = 0
            self._open()

    def close(self):
        if self._file:
            self._file.close()


async def scrape_database(
    db_config: dict, state: dict, session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore, progress: Progress, task_id,
):
    db_name = db_config["db"]
    db_short = db_config["short"]
    scraped_ids: set[str] = set(state["scraped_doc_ids"].get(db_name, []))
    progress.update(task_id, completed=len(scraped_ids))

    writer = ChunkWriter(db_short)
    start_url = f"{BASE_URL}/{db_name}/{db_config['view_id']}?OpenView&Start=1"
    current_url = state["db_progress"].get(db_name, start_url)

    async def fetch_doc(url: str):
        async with semaphore:
            content = await fetch_with_retry(session, url)
        if not content:
            return None
        return parse_document(decode_content(content), url, db_config)

    while current_url:
        async with semaphore:
            doc_urls, next_url = await get_doc_links_from_page(session, current_url, db_config)

        new_urls = [u for u in doc_urls
                    if u.rstrip("/").split("/")[-1].split("?")[0] not in scraped_ids]
        if new_urls:
            results = await asyncio.gather(
                *[fetch_doc(u) for u in new_urls], return_exceptions=True
            )
            for res in results:
                if not res or isinstance(res, Exception):
                    continue
                doc_id = res["doc_id"]
                if doc_id and doc_id not in scraped_ids:
                    writer.write(res)
                    scraped_ids.add(doc_id)
                    progress.advance(task_id)

        with _state_lock:
            state["scraped_doc_ids"][db_name] = list(scraped_ids)
            state["db_progress"][db_name] = next_url or current_url
            save_state(state)

        if not next_url:
            break
        current_url = next_url

    writer.close()
    with _state_lock:
        if db_name not in state["completed_dbs"]:
            state["completed_dbs"].append(db_name)
        save_state(state)


async def run_all(
    dbs_to_scrape: list[dict],
    state: dict,
    concurrency: int,
    progress: Progress,
):
    """One shared aiohttp session + global semaphore; every DB runs in parallel."""
    connector = aiohttp.TCPConnector(limit=concurrency, ssl=False, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=60)
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(
        headers=HEADERS, connector=connector, timeout=timeout
    ) as session:
        tasks = []
        for db_config in dbs_to_scrape:
            task_id = progress.add_task(
                f"{db_config['short']:5s} {db_config['label'][:38]}",
                total=db_config["approx_count"],
                completed=len(state["scraped_doc_ids"].get(db_config["db"], [])),
            )
            tasks.append(scrape_database(
                db_config, state, session, semaphore, progress, task_id,
            ))
        await asyncio.gather(*tasks)


@click.command()
@click.option("--db", default=None, help="Only scrape specific DB (short name)")
@click.option("--reset", is_flag=True, help="Reset state and start from scratch")
@click.option("--concurrency", default=80, type=int,
              help="Global max concurrent HTTP connections across all DBs")
def main(db: Optional[str], reset: bool, concurrency: int):
    """DGSI caselaw scraper — fetches Portuguese court decisions in parallel."""
    if reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        console.print("[yellow]State reset.[/yellow]")

    state = load_state()
    state.setdefault("completed_dbs", [])
    state.setdefault("scraped_doc_ids", {})
    state.setdefault("db_progress", {})

    dbs_to_scrape = DATABASES
    if db:
        dbs_to_scrape = [d for d in DATABASES if d["short"].upper() == db.upper()]
        if not dbs_to_scrape:
            console.print(f"[red]Unknown database: {db}[/red]")
            sys.exit(1)

    dbs_to_scrape = [d for d in dbs_to_scrape if d["db"] not in state["completed_dbs"]]
    if not dbs_to_scrape:
        console.print("[green]All databases already scraped.[/green]")
        return

    total_approx = sum(d["approx_count"] for d in dbs_to_scrape)
    console.print(Panel.fit(
        f"[bold]DGSI Scraper[/bold]\n"
        f"Databases: {len(dbs_to_scrape)} in parallel\n"
        f"Approx documents: {total_approx:,}\n"
        f"Global concurrency: {concurrency}",
        title="Starting",
    ))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        asyncio.run(run_all(dbs_to_scrape, state, concurrency, progress))

    console.print("[bold green]Scraping complete.[/bold green]")


if __name__ == "__main__":
    main()
