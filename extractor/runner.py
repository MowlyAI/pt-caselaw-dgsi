"""
Entry point for the LLM extractor.
Run: python -m extractor.runner [--db STJ] [--reset]
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click
import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, ProgressColumn, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.text import Text


class RateColumn(ProgressColumn):
    """Render live throughput in docs/sec using Rich's auto-computed speed."""

    def render(self, task) -> Text:
        if task.speed is None:
            return Text("-- docs/s", style="dim")
        return Text(f"{task.speed:.1f} docs/s", style="cyan")

load_dotenv(".env.local")
sys.path.insert(0, str(Path(__file__).parent.parent))
from extractor.extractor import extract_document
from scraper.config import DATABASES

console = Console()

RAW_DATA_DIR = Path("data/raw")
ENHANCED_DIR = Path("data/enhanced")
STATE_FILE = Path("data/extractor_state.json")
DOCS_PER_CHUNK = 15000


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Corrupt state (e.g. SIGKILL mid-write on a pre-atomic version).
            # Fall back to empty — the dedup set just repopulates from scratch
            # and Supabase upsert dedups on doc_id anyway.
            console.print(f"[yellow]WARN: {STATE_FILE} is corrupt — starting from empty state[/yellow]")
    return {"processed_doc_ids": {}}


def save_state(state: dict):
    """Persist state to disk.

    Write directly (non-atomic) with fsync to ensure state survives
    a SIGKILL. The state file is only read at startup, so minor
    temporary inconsistencies are safe — worst case we reprocess a
    few docs on restart."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Write directly to avoid os.replace() issues on macOS APFS
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())


def iter_raw_docs(db_short: str):
    """Iterate over all raw documents for a database."""
    db_dir = RAW_DATA_DIR / db_short
    if not db_dir.exists():
        return
    for chunk_file in sorted(db_dir.glob("chunk_*.jsonl")):
        with open(chunk_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


class EnhancedWriter:
    def __init__(self, db_short: str):
        self.db_short = db_short
        self.chunk = 0
        self.count = 0
        self._file = None
        self._open()

    def _open(self):
        if self._file:
            self._file.close()
        path = ENHANCED_DIR / self.db_short / f"chunk_{self.chunk:04d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    def write(self, doc: dict):
        self._file.write(json.dumps(doc, ensure_ascii=False) + "\n")
        # flush Python's buffer so a SIGKILL never loses recently-written
        # extractions (OS page cache survives process kill).
        self._file.flush()
        self.count += 1
        if self.count >= DOCS_PER_CHUNK:
            self.chunk += 1
            self.count = 0
            self._open()

    def close(self):
        if self._file:
            self._file.close()


async def _extract_one(
    client: httpx.AsyncClient,
    doc: dict,
    api_key: str,
    semaphore: asyncio.Semaphore,
):
    """Wrap extract_document so as_completed sees (doc, result) tuples."""
    try:
        result = await extract_document(client, doc, api_key, semaphore)
        return doc, result, None
    except Exception as e:
        return doc, None, e


async def _run_batch(
    client: httpx.AsyncClient,
    batch: list,
    api_key: str,
    semaphore: asyncio.Semaphore,
    writer: "EnhancedWriter",
    processed_ids: set,
    progress: Progress,
    task_id,
    stats: dict,
) -> None:
    """Extract a batch concurrently and stream-process results as they land.
    Successful docs are written to the enhanced chunk and added to
    processed_ids; failed docs stay unprocessed so the next daily run retries
    them automatically. Results are processed in completion order (not input
    order) so the progress bar updates immediately — even if one slow/retried
    request would otherwise hold up a whole batch."""
    tasks = [
        asyncio.create_task(_extract_one(client, d, api_key, semaphore))
        for d in batch
    ]
    batch_failed = 0
    for coro in asyncio.as_completed(tasks):
        doc, result, _err = await coro
        doc_id = doc.get("doc_id", "")
        if result is None:
            batch_failed += 1
            stats["fail"] += 1
            progress.advance(task_id)
            continue
        writer.write({**doc, "llm_extracted": result})
        if doc_id:
            processed_ids.add(doc_id)
        stats["ok"] += 1
        if result.get("is_jurisprudence_unification"):
            stats["auj"] += 1
        progress.advance(task_id)
        progress.update(task_id, ok=stats["ok"], fail=stats["fail"], auj=stats["auj"])
    if batch_failed:
        console.print(
            f"[yellow]{batch_failed}/{len(batch)} failed in batch "
            f"(cumulative: ok={stats['ok']} fail={stats['fail']}); will retry on next run[/yellow]"
        )


async def process_database(
    db_config: dict,
    state: dict,
    api_key: str,
    concurrency: int,
    progress: Progress,
    task_id,
):
    db_name = db_config["db"]
    db_short = db_config["short"]

    processed_ids = set(state.get("processed_doc_ids", {}).get(db_name, []))
    writer = EnhancedWriter(db_short)
    stats = {"ok": 0, "fail": 0, "auj": 0, "t0": time.perf_counter(), "last_log": 0}
    total_target = db_config["approx_count"]

    semaphore = asyncio.Semaphore(concurrency)
    # httpx client with connection pool sized for the concurrency level so
    # we don't serialise requests behind the default 10-connection limit.
    limits = httpx.Limits(
        max_connections=concurrency + 20,
        max_keepalive_connections=concurrency,
    )

    async with httpx.AsyncClient(limits=limits) as client:
        batch = []
        batch_size = concurrency * 2

        for doc in iter_raw_docs(db_short):
            doc_id = doc.get("doc_id", "")
            if doc_id in processed_ids:
                progress.advance(task_id)
                continue
            batch.append(doc)

            if len(batch) >= batch_size:
                await _run_batch(
                    client, batch, api_key, semaphore, writer,
                    processed_ids, progress, task_id, stats,
                )
                state["processed_doc_ids"][db_name] = list(processed_ids)
                save_state(state)
                batch = []

                # Log-friendly heartbeat every ~30s for log-tailing usage
                # (the live Rich progress bar refreshes continuously anyway).
                now = time.perf_counter()
                if now - stats["last_log"] >= 30:
                    stats["last_log"] = now
                    done = stats["ok"] + stats["fail"]
                    elapsed = now - stats["t0"]
                    rate = done / elapsed if elapsed else 0
                    remaining = max(0, total_target - len(processed_ids))
                    eta_s = remaining / rate if rate > 0 else 0
                    eta_h = eta_s / 3600
                    succ = stats["ok"] / done * 100 if done else 100
                    console.print(
                        f"[blue][heartbeat {db_short}][/blue] "
                        f"processed={len(processed_ids)}/{total_target} "
                        f"ok={stats['ok']} fail={stats['fail']} auj={stats['auj']} "
                        f"success={succ:.1f}% rate={rate:.1f}/s ETA={eta_h:.1f}h"
                    )

        # Process remaining batch
        if batch:
            await _run_batch(
                client, batch, api_key, semaphore, writer,
                processed_ids, progress, task_id, stats,
            )
            state["processed_doc_ids"][db_name] = list(processed_ids)
            save_state(state)

    writer.close()
    total = stats["ok"] + stats["fail"]
    rate = stats["ok"] / total * 100 if total else 0.0
    console.print(
        f"[green]✓ Extracted: {db_config['label']} — "
        f"ok={stats['ok']} fail={stats['fail']} success={rate:.2f}% "
        f"auj={stats['auj']} | total state: {len(processed_ids)} docs[/green]"
    )


@click.command()
@click.option("--db", default=None, help="Only process specific DB (short name)")
@click.option("--reset", is_flag=True, help="Reset extractor state")
@click.option("--concurrency", default=20, help="Concurrent LLM requests")
def main(db: Optional[str], reset: bool, concurrency: int):
    """Extract structured info from scraped documents using LLM."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not set in .env.local[/red]")
        sys.exit(1)

    if reset and STATE_FILE.exists():
        STATE_FILE.unlink()

    state = load_state()
    dbs = DATABASES
    if db:
        dbs = [d for d in DATABASES if d["short"].upper() == db.upper()]

    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)

    total_counts = {d["short"]: d["approx_count"] for d in dbs}

    with Progress(
        SpinnerColumn(), TextColumn("[bold]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TaskProgressColumn(),
        TextColumn(
            "[green]✓{task.fields[ok]}[/green] "
            "[red]✗{task.fields[fail]}[/red] "
            "[magenta]auj:{task.fields[auj]}[/magenta]"
        ),
        RateColumn(),
        TimeElapsedColumn(), TimeRemainingColumn(),
        console=console, refresh_per_second=2,
    ) as progress:
        for db_config in dbs:
            processed = len(state.get("processed_doc_ids", {}).get(db_config["db"], []))
            task_id = progress.add_task(
                f"Extract {db_config['short']}",
                total=db_config["approx_count"],
                completed=processed,
                ok=0, fail=0, auj=0,
            )
            asyncio.run(process_database(db_config, state, api_key, concurrency, progress, task_id))

    console.print("[bold green]Extraction complete![/bold green]")


if __name__ == "__main__":
    main()
