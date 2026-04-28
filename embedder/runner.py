"""Entry point for the embedder. Generates embeddings + uploads to Supabase."""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import click
import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

load_dotenv(".env.local")
sys.path.insert(0, str(Path(__file__).parent.parent))
from embedder.embedder import (
    build_embedding_texts, doc_to_row, generate_embeddings_batch, BATCH_SIZE as EMB_BATCH_SIZE,
)
from scraper.config import DATABASES

console = Console()

ENHANCED_DIR = Path("data/enhanced")
STATE_FILE = Path("data/embedder_state.json")


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            console.print(f"[yellow]WARN: {STATE_FILE} is corrupt — starting from empty state[/yellow]")
    return {"uploaded_doc_ids": {}}


def save_state(state: dict):
    """Direct state write with fsync (no atomic rename — safe for single-writer)."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())


def iter_enhanced_docs(db_short: str):
    db_dir = ENHANCED_DIR / db_short
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


async def upsert_to_supabase(
    client: httpx.AsyncClient,
    supabase_url: str,
    supabase_key: str,
    rows: list[dict],
) -> bool:
    """Upsert a batch of documents to Supabase."""
    if not rows:
        return True
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    url = f"{supabase_url}/rest/v1/documents?on_conflict=doc_id"
    for attempt in range(5):
        try:
            resp = await client.post(url, json=rows, headers=headers, timeout=60)
            if resp.status_code in (200, 201, 204):
                return True
            if resp.status_code in (429, 503, 500):
                await asyncio.sleep(2 ** attempt)
                continue
            console.print(f"[red]Supabase upsert error {resp.status_code}: {resp.text[:300]}[/red]")
            return False
        except (httpx.TimeoutException, httpx.RequestError) as e:
            if attempt == 4:
                console.print(f"[red]Supabase upsert failed: {e}[/red]")
                return False
            await asyncio.sleep(2 ** attempt)
    return False


async def process_database(
    db_config: dict, state: dict, api_key: str,
    supabase_url: str, supabase_key: str,
    concurrency: int, progress: Progress, task_id,
):
    db_name = db_config["db"]
    db_short = db_config["short"]
    uploaded_ids = set(state.get("uploaded_doc_ids", {}).get(db_name, []))
    semaphore = asyncio.Semaphore(concurrency)

    limits = httpx.Limits(max_connections=concurrency + 50, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(300, connect=30)) as client:
        batch_docs = []
        # Large batches — each batch makes ceil(N/100)*3 API calls
        # 500 docs → 15 API calls (5 chunks × 3 fields)
        batch_size = 500

        BATCH_TIMEOUT = 600  # 10 min max per batch of 500 docs

        for doc in iter_enhanced_docs(db_short):
            doc_id = doc.get("doc_id", "")
            if not doc_id or doc_id in uploaded_ids:
                progress.advance(task_id)
                continue
            batch_docs.append(doc)
            if len(batch_docs) >= batch_size:
                try:
                    await asyncio.wait_for(
                        _process_batch(client, batch_docs, api_key, supabase_url, supabase_key, semaphore, uploaded_ids, progress, task_id),
                        timeout=BATCH_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    console.print(f"  [red]{db_short}: batch timed out after {BATCH_TIMEOUT}s, skipping[/red]", highlight=False)
                except Exception as e:
                    console.print(f"  [red]{db_short}: batch error: {e}[/red]", highlight=False)
                state["uploaded_doc_ids"][db_name] = list(uploaded_ids)
                save_state(state)
                console.print(f"  {db_short}: {len(uploaded_ids):,} uploaded | batch done", highlight=False)
                batch_docs = []
        if batch_docs:
            try:
                await asyncio.wait_for(
                    _process_batch(client, batch_docs, api_key, supabase_url, supabase_key, semaphore, uploaded_ids, progress, task_id),
                    timeout=BATCH_TIMEOUT,
                )
            except (asyncio.TimeoutError, Exception) as e:
                console.print(f"  [red]{db_short}: final batch error: {e}[/red]", highlight=False)
            state["uploaded_doc_ids"][db_name] = list(uploaded_ids)
            save_state(state)

    console.print(f"✓ Embedded & uploaded: {db_config['label']} — {len(uploaded_ids):,} docs | total state: {sum(len(v) for v in state.get('uploaded_doc_ids', {}).values()):,} docs", highlight=False)


EMBED_FIELDS = ["embedding", "embedding_context", "embedding_ratio"]


async def _process_batch(client, batch_docs, api_key, supabase_url, supabase_key, semaphore, uploaded_ids, progress, task_id):
    """Generate 3 independent embeddings per doc using batch API calls, then upsert.

    Instead of N×3 individual API calls, we make ceil(N/100)×3 batch calls.
    For a batch of 500 docs: 5×3 = 15 API calls instead of 1500.
    """
    # Build text inputs for each embedding field
    all_texts: dict[str, list[str]] = {f: [] for f in EMBED_FIELDS}
    for doc in batch_docs:
        texts = build_embedding_texts(doc)
        for field in EMBED_FIELDS:
            all_texts[field].append(texts[field])

    # Generate embeddings using batch API — 3 concurrent streams, each chunked
    async def _embed_field(field: str, texts: list[str]) -> list[Optional[list[float]]]:
        """Embed all texts for one field using chunked batch requests."""
        results: list[Optional[list[float]]] = [None] * len(texts)
        tasks = []
        for chunk_start in range(0, len(texts), EMB_BATCH_SIZE):
            chunk = texts[chunk_start:chunk_start + EMB_BATCH_SIZE]
            # Filter empty texts but track their positions
            non_empty_indices = [i for i, t in enumerate(chunk) if t.strip()]
            non_empty_texts = [chunk[i] for i in non_empty_indices]
            if not non_empty_texts:
                continue
            tasks.append((chunk_start, non_empty_indices, non_empty_texts))

        # Fire all chunk requests for this field in parallel
        batch_coros = [
            generate_embeddings_batch(client, ne_texts, api_key, semaphore)
            for _, _, ne_texts in tasks
        ]
        batch_results = await asyncio.gather(*batch_coros, return_exceptions=True)

        for (chunk_start, ne_indices, _), batch_embs in zip(tasks, batch_results):
            if isinstance(batch_embs, Exception) or batch_embs is None:
                continue
            for local_idx, emb in zip(ne_indices, batch_embs):
                results[chunk_start + local_idx] = emb
        return results

    # Run all 3 fields in parallel
    field_results = await asyncio.gather(*[
        _embed_field(field, all_texts[field]) for field in EMBED_FIELDS
    ])
    embeddings_by_field = dict(zip(EMBED_FIELDS, field_results))

    # Build rows
    good_rows: list[dict] = []
    good_doc_ids: list[str] = []
    failed = 0
    for i, doc in enumerate(batch_docs):
        progress.advance(task_id)
        primary = embeddings_by_field["embedding"][i]
        if primary is None:
            failed += 1
            continue
        row = doc_to_row(doc)
        for field in EMBED_FIELDS:
            row[field] = embeddings_by_field[field][i]
        good_rows.append(row)
        good_doc_ids.append(doc.get("doc_id", ""))

    # Upsert in chunks (Supabase has payload size limits)
    UPSERT_CHUNK = 50
    all_ok = True
    for j in range(0, len(good_rows), UPSERT_CHUNK):
        chunk_rows = good_rows[j:j + UPSERT_CHUNK]
        chunk_ids = good_doc_ids[j:j + UPSERT_CHUNK]
        ok = await upsert_to_supabase(client, supabase_url, supabase_key, chunk_rows)
        if ok:
            uploaded_ids.update(d for d in chunk_ids if d)
        else:
            all_ok = False

    if failed:
        console.print(f"[yellow]{failed}/{len(batch_docs)} embeddings failed[/yellow]")


@click.command()
@click.option("--db", default=None, help="Only process specific DB (short name)")
@click.option("--reset", is_flag=True, help="Reset embedder state")
@click.option("--concurrency", default=20, help="Concurrent embedding requests")
def main(db: Optional[str], reset: bool, concurrency: int):
    """Generate embeddings and upload to Supabase."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not api_key or not supabase_url or not supabase_key:
        console.print("[red]Missing env vars: OPENROUTER_API_KEY / SUPABASE_URL / SUPABASE_SERVICE_KEY[/red]")
        sys.exit(1)

    if reset and STATE_FILE.exists():
        STATE_FILE.unlink()

    state = load_state()
    dbs = DATABASES
    if db:
        dbs = [d for d in DATABASES if d["short"].upper() == db.upper()]

    with Progress(
        SpinnerColumn(), TextColumn("[bold]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TaskProgressColumn(),
        TimeElapsedColumn(), TimeRemainingColumn(),
        console=console, refresh_per_second=2,
    ) as progress:
        for db_config in dbs:
            uploaded = len(state.get("uploaded_doc_ids", {}).get(db_config["db"], []))
            task_id = progress.add_task(
                f"Embed {db_config['short']}",
                total=db_config["approx_count"], completed=uploaded,
            )
            asyncio.run(process_database(
                db_config, state, api_key, supabase_url, supabase_key,
                concurrency, progress, task_id,
            ))

    console.print("[bold green]Embedding & upload complete![/bold green]")


if __name__ == "__main__":
    main()
