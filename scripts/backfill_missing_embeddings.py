"""Fill in missing embedding_context / embedding_ratio for existing documents.

The embedder only treats a document as failed when its primary embedding is
missing, so transient API failures on the two secondary fields were never
retried. This rebuilds the embedding inputs from each row's stored metadata
(the LLM extraction) and updates only the NULL fields.

Usage:
    python scripts/backfill_missing_embeddings.py            # all rows
    python scripts/backfill_missing_embeddings.py --dry-run  # count only
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
import click
import httpx
from dotenv import load_dotenv

load_dotenv(".env.local")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from embedder.embedder import BATCH_SIZE, build_embedding_texts, generate_embeddings_batch

FIELDS = ["embedding_context", "embedding_ratio"]


async def run(dry_run: bool) -> None:
    conn = await asyncpg.connect(
        host=os.getenv("SUPABASE_DB_HOST"),
        port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
        user=os.getenv("SUPABASE_DB_USER"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        database=os.getenv("SUPABASE_DB_NAME", "postgres"),
        statement_cache_size=0,
        command_timeout=600,
    )
    await conn.execute("SET statement_timeout = '600s'")
    rows = await conn.fetch(
        "SELECT doc_id, metadata::text AS metadata, "
        "embedding_context IS NULL AS need_context, embedding_ratio IS NULL AS need_ratio "
        "FROM documents WHERE embedding IS NOT NULL "
        "AND (embedding_context IS NULL OR embedding_ratio IS NULL)"
    )
    todo = []  # (doc_id, field, text)
    empty = 0
    for r in rows:
        texts = build_embedding_texts({"llm_extracted": json.loads(r["metadata"])})
        for field, needed in (("embedding_context", r["need_context"]), ("embedding_ratio", r["need_ratio"])):
            if not needed:
                continue
            if texts[field].strip():
                todo.append((r["doc_id"], field, texts[field]))
            else:
                empty += 1
    print(f"{len(rows)} documents with a missing secondary embedding: "
          f"{len(todo)} fields to embed, {empty} have no source text (left NULL)", flush=True)
    if dry_run or not todo:
        await conn.close()
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    sem = asyncio.Semaphore(10)
    done = failed = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30)) as client:
        for start in range(0, len(todo), BATCH_SIZE):
            chunk = todo[start:start + BATCH_SIZE]
            vectors = await generate_embeddings_batch(client, [t for _, _, t in chunk], api_key, sem)
            if not vectors:
                failed += len(chunk)
                continue
            for field in FIELDS:
                updates = [(json.dumps(v), doc_id) for (doc_id, f, _), v in zip(chunk, vectors) if f == field and v]
                if updates:
                    await conn.executemany(
                        f"UPDATE documents SET {field} = $1::text::halfvec WHERE doc_id = $2", updates
                    )
                    done += len(updates)
            print(f"  {done}/{len(todo)} fields updated ({failed} failed)", flush=True)
    await conn.close()
    print(f"Done: {done} updated, {failed} failed")


@click.command()
@click.option("--dry-run", is_flag=True, help="Only report what would be embedded")
def main(dry_run: bool) -> None:
    asyncio.run(run(dry_run))


if __name__ == "__main__":
    main()
