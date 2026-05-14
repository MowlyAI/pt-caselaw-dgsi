import asyncio
import os
import time
from typing import Any

import asyncpg
import httpx
from dotenv import load_dotenv

load_dotenv('.env.local')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE = 'https://openrouter.ai/api/v1'
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'google/gemini-embedding-001')
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIMENSION', '1024'))
BATCH_SIZE = 50   # texts per embedding request
CONCURRENCY = 1   # serial to avoid Gemini rate limits
SLEEP_BETWEEN_BATCHES = 1.0  # seconds


def _citation_text(law: str, article: str | None) -> str:
    if article and article.strip():
        return f"{law} art. {article.strip()}"
    return law.strip()


async def _fetch_batches(conn: asyncpg.Connection):
    """Yield batches of (law, article, citation_text, doc_count) from DB."""
    # For testing, add LIMIT 100; remove for full run
    rows = await conn.fetch(
        """
        SELECT l->>'law' as law, l->>'article' as article, COUNT(*) as doc_count
        FROM documents, jsonb_array_elements(metadata->'legislation_cited') AS l
        WHERE l->>'law' IS NOT NULL AND l->>'law' != ''
        GROUP BY l->>'law', l->>'article'
        LIMIT 100
        """
    )
    print(f'Fetched {len(rows)} unique legislation rows from DB.')

    batch: list[tuple[str, str | None, str, int]] = []
    for r in rows:
        law = r['law']
        article = r['article'] if r['article'] and r['article'].strip() else None
        ct = _citation_text(law, article)
        batch.append((law, article, ct, r['doc_count']))
        if len(batch) >= BATCH_SIZE:
            yield batch
            batch = []
    if batch:
        yield batch


async def _embed_batch(texts: list[str], sem: asyncio.Semaphore) -> list[list[float]]:
    """Call OpenRouter embedding API with a batch of texts. Retries on transient errors."""
    if not texts:
        return []
    async with sem:
        async with httpx.AsyncClient(timeout=120) as client:
            for attempt in range(5):
                resp = await client.post(
                    f'{OPENROUTER_BASE}/embeddings',
                    json={'model': EMBEDDING_MODEL, 'input': texts, 'dimensions': EMBEDDING_DIM},
                    headers={'Authorization': f'Bearer {OPENROUTER_API_KEY}'},
                )
                if resp.status_code == 200:
                    body = resp.json()
                    if 'data' not in body:
                        raise RuntimeError(f'Unexpected response (no data): {resp.text[:300]}')
                    data = body['data']
                    if len(data) != len(texts):
                        raise RuntimeError(f'Batch size mismatch: sent {len(texts)}, got {len(data)}')
                    return [d['embedding'] for d in data]
                # Rate limit / transient error
                if resp.status_code in (429, 502, 503, 504):
                    wait = 2 ** attempt
                    print(f'  Embedding API {resp.status_code}, retrying in {wait}s...')
                    await asyncio.sleep(wait)
                    continue
                raise RuntimeError(f'Embedding API error: {resp.status_code} {resp.text[:300]}')
            raise RuntimeError('Embedding API max retries exceeded')


async def _embed_one_batch(
    batch: list[tuple[str, str | None, str, int]],
    sem: asyncio.Semaphore,
) -> list[tuple[str, str | None, str, int, str]] | None:
    """Embed a batch and return values ready for insert."""
    texts = [item[2] for item in batch]
    try:
        embeddings = await _embed_batch(texts, sem)
    except Exception as e:
        print(f'Batch failed: {e}')
        return None

    def _emb_str(emb: list[float]) -> str:
        return '[' + ','.join(str(v) for v in emb) + ']'

    return [
        (law, article, ct, doc_count, _emb_str(embedding))
        for (law, article, ct, doc_count), embedding in zip(batch, embeddings)
    ]


async def main():
    conn = await asyncpg.connect(
        host=os.getenv('SUPABASE_DB_HOST'),
        port=int(os.getenv('SUPABASE_DB_PORT', '5432')),
        user=os.getenv('SUPABASE_DB_USER'),
        password=os.getenv('SUPABASE_DB_PASSWORD'),
        database=os.getenv('SUPABASE_DB_NAME', 'postgres'),
    )

    existing = await conn.fetchval('SELECT COUNT(*) FROM legislation_embeddings')
    print(f'Existing legislation_embeddings rows: {existing}')

    # Use temp table to reliably extract all unique legislation
    print('Extracting unique legislation via temp table...')
    await conn.execute("SET statement_timeout = '300000'")
    await conn.execute("DROP TABLE IF EXISTS tmp_leg_uniq")
    await conn.execute("""
        CREATE TEMP TABLE tmp_leg_uniq AS
        SELECT DISTINCT l->>'law' as law, l->>'article' as article
        FROM documents, jsonb_array_elements(metadata->'legislation_cited') AS l
        WHERE l->>'law' IS NOT NULL AND l->>'law' != ''
    """)
    total_uniq = await conn.fetchval("SELECT COUNT(*) FROM tmp_leg_uniq")
    print(f'Total unique legislation pairs: {total_uniq}')

    # Delete already-embedded rows from temp table for resumability
    await conn.execute("""
        DELETE FROM tmp_leg_uniq t
        WHERE EXISTS (
            SELECT 1 FROM legislation_embeddings e
            WHERE e.law = t.law AND (e.article IS NOT DISTINCT FROM t.article)
        )
    """)
    remaining = await conn.fetchval("SELECT COUNT(*) FROM tmp_leg_uniq")
    print(f'Remaining to embed: {remaining}')

    PAGE_SIZE = 5000
    all_batches: list[list[tuple[str, str | None, str, int]]] = []
    offset = 0
    while offset < remaining:
        rows = await conn.fetch(
            f"SELECT law, article FROM tmp_leg_uniq ORDER BY law, article LIMIT {PAGE_SIZE} OFFSET {offset}"
        )
        if not rows:
            break
        batch: list[tuple[str, str | None, str, int]] = []
        for r in rows:
            law = r['law']
            article = r['article'] if r['article'] and r['article'].strip() else None
            ct = _citation_text(law, article)
            batch.append((law, article, ct, 0))
            if len(batch) >= BATCH_SIZE:
                all_batches.append(batch)
                batch = []
        if batch:
            all_batches.append(batch)
        offset += PAGE_SIZE
        print(f'  Prepared {len(all_batches)} batches so far (offset {offset})')
    print(f'Total batches to process: {len(all_batches)}')

    total_inserted = [0]
    start = time.time()
    sem = asyncio.Semaphore(CONCURRENCY)

    for i, batch in enumerate(all_batches):
        db_values = await _embed_one_batch(batch, sem)
        if db_values:
            await conn.executemany(
                """
                INSERT INTO legislation_embeddings (law, article, citation_text, doc_count, embedding)
                VALUES ($1, $2, $3, $4, $5::halfvec)
                ON CONFLICT (law, article) DO UPDATE SET
                    citation_text = EXCLUDED.citation_text,
                    doc_count = EXCLUDED.doc_count,
                    embedding = EXCLUDED.embedding
                """,
                db_values,
            )
            total_inserted[0] += len(db_values)
            elapsed = time.time() - start
            print(f'Inserted {total_inserted[0]} rows, {elapsed:.1f}s elapsed')
        if i < len(all_batches) - 1:
            await asyncio.sleep(SLEEP_BETWEEN_BATCHES)

    print(f'Done. Total inserted/updated: {total_inserted[0]}')
    await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
