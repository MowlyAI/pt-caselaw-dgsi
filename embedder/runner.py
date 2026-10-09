"""Entry point for the embedder. Generates embeddings + uploads to Supabase."""
import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import asyncpg
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


UPSERT_COLUMNS = [
    "doc_id", "url", "court_short", "decision_date", "process_number", "legal_domain",
    "is_auj", "full_text", "summary", "metadata",
    "embedding", "embedding_context", "embedding_ratio",
]
_CASTS = {"decision_date": "::date", "metadata": "::jsonb",
          "embedding": "::text::halfvec", "embedding_context": "::text::halfvec",
          "embedding_ratio": "::text::halfvec"}
UPSERT_SQL = (
    f"INSERT INTO documents ({', '.join(UPSERT_COLUMNS)}) VALUES ("
    + ", ".join(f"${i + 1}{_CASTS.get(c, '')}" for i, c in enumerate(UPSERT_COLUMNS))
    + ") ON CONFLICT (doc_id) DO UPDATE SET "
    + ", ".join(f"{c} = EXCLUDED.{c}" for c in UPSERT_COLUMNS[1:])
)


async def create_db_pool() -> asyncpg.Pool:
    """Direct Postgres pool. Bypasses PostgREST, whose authenticator role caps
    statements at 8s — too short now that each row updates three HNSW indexes."""
    return await asyncpg.create_pool(
        host=os.getenv("SUPABASE_DB_HOST"),
        port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
        user=os.getenv("SUPABASE_DB_USER"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        database=os.getenv("SUPABASE_DB_NAME", "postgres"),
        min_size=1, max_size=UPSERT_CONCURRENCY,
        statement_cache_size=0,
        command_timeout=600,
        init=lambda conn: conn.execute("SET statement_timeout = '600s'"),
    )


def _row_values(row: dict) -> tuple:
    out = []
    for col in UPSERT_COLUMNS:
        v = row.get(col)
        if col == "decision_date" and v:
            v = date.fromisoformat(v)
        elif col == "metadata":
            v = json.dumps(v or {}, ensure_ascii=False)
        elif col.startswith("embedding") and v is not None:
            v = json.dumps(v)
        out.append(v)
    return tuple(out)


async def upsert_rows(pool: asyncpg.Pool, rows: list[dict]) -> bool:
    """Upsert a chunk of documents in one transaction, retrying transient errors."""
    if not rows:
        return True
    values = [_row_values(r) for r in rows]
    for attempt in range(5):
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(UPSERT_SQL, values)
            return True
        except (asyncpg.PostgresError, OSError, asyncio.TimeoutError) as e:
            if attempt == 4:
                console.print(f"[red]Postgres upsert failed: {e}[/red]")
                return False
            await asyncio.sleep(2 ** attempt)
    return False


async def process_database(
    db_config: dict, state: dict, api_key: str,
    concurrency: int, progress: Progress, task_id,
    only_ids: Optional[set[str]] = None,
):
    db_name = db_config["db"]
    db_short = db_config["short"]
    uploaded_ids = set(state.get("uploaded_doc_ids", {}).get(db_name, []))
    semaphore = asyncio.Semaphore(concurrency)

    limits = httpx.Limits(max_connections=concurrency + 50, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(300, connect=30)) as client, \
            await create_db_pool() as pool:
        batch_docs = []
        # 100 docs → 3 embedding API calls; small enough that state is saved
        # every few minutes given the slow upserts (~0.6 docs/s on HNSW).
        batch_size = 100
        BATCH_TIMEOUT = 3600

        for doc in iter_enhanced_docs(db_short):
            doc_id = doc.get("doc_id", "")
            if not doc_id or doc_id in uploaded_ids or (only_ids is not None and doc_id not in only_ids):
                progress.advance(task_id)
                continue
            batch_docs.append(doc)
            if len(batch_docs) >= batch_size:
                try:
                    await asyncio.wait_for(
                        _process_batch(client, batch_docs, api_key, pool, semaphore, uploaded_ids, progress, task_id),
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
                    _process_batch(client, batch_docs, api_key, pool, semaphore, uploaded_ids, progress, task_id),
                    timeout=BATCH_TIMEOUT,
                )
            except (asyncio.TimeoutError, Exception) as e:
                console.print(f"  [red]{db_short}: final batch error: {e}[/red]", highlight=False)
            state["uploaded_doc_ids"][db_name] = list(uploaded_ids)
            save_state(state)

    console.print(f"✓ Embedded & uploaded: {db_config['label']} — {len(uploaded_ids):,} docs | total state: {sum(len(v) for v in state.get('uploaded_doc_ids', {}).values()):,} docs", highlight=False)


EMBED_FIELDS = ["embedding", "embedding_context", "embedding_ratio"]
UPSERT_CONCURRENCY = 4


async def _process_batch(client, batch_docs, api_key, pool, semaphore, uploaded_ids, progress, task_id):
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

    # Each row costs ~1-6s (3 HNSW indexes + FTS trigger on full_text), so keep
    # transactions small; UPSERT_CONCURRENCY chunks run in parallel.
    UPSERT_CHUNK = 10

    async def _upsert_chunk(chunk_rows: list[dict], chunk_ids: list[str]):
        ok = await upsert_rows(pool, chunk_rows)
        if ok:
            uploaded_ids.update(d for d in chunk_ids if d)

    await asyncio.gather(*[
        _upsert_chunk(good_rows[j:j + UPSERT_CHUNK], good_doc_ids[j:j + UPSERT_CHUNK])
        for j in range(0, len(good_rows), UPSERT_CHUNK)
    ])

    if failed:
        console.print(f"[yellow]{failed}/{len(batch_docs)} embeddings failed[/yellow]")


@click.command()
@click.option("--db", default=None, help="Only process specific DB (short name)")
@click.option("--reset", is_flag=True, help="Reset embedder state")
@click.option("--concurrency", default=20, help="Concurrent embedding requests")
@click.option("--ids-file", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Only upload doc_ids listed in this file (one per line)")
def main(db: Optional[str], reset: bool, concurrency: int, ids_file: Optional[str]):
    """Generate embeddings and upload to Supabase."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    missing = [v for v in ("OPENROUTER_API_KEY", "SUPABASE_DB_HOST", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD")
               if not os.getenv(v)]
    if missing:
        console.print(f"[red]Missing env vars: {', '.join(missing)}[/red]")
        sys.exit(1)

    if reset and STATE_FILE.exists():
        STATE_FILE.unlink()

    only_ids = None
    if ids_file:
        only_ids = {line.strip() for line in open(ids_file) if line.strip()}
        console.print(f"Restricting upload to {len(only_ids):,} doc_ids from {ids_file}")

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
                db_config, state, api_key, concurrency, progress, task_id, only_ids,
            ))

    console.print("[bold green]Embedding & upload complete![/bold green]")


if __name__ == "__main__":
    main()
