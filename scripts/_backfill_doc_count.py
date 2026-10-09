"""Backfill doc_count in legislation_embeddings.

Strategy: count citing documents per (law, article) once from the narrow
document_legislation_citations table into a temp table, then update
legislation_embeddings in committed id-range batches. The previous per-law
approach scanned TOASTed documents.metadata once per law and never finished
on a small instance (~72% of rows were left at doc_count = 0).

Batches matter: heap pages are full, so updates are not HOT and every changed
row is re-inserted into all indexes, including the HNSW index. A single
statement over ~260k rows risks hitting the timeout and rolling back.
Rows with no match in the citations table keep their existing value.
"""
import asyncio
import os
import time

import asyncpg
from dotenv import load_dotenv

load_dotenv('.env.local')

BATCH_IDS = 10_000

COUNTS_SQL = """
CREATE TEMP TABLE tmp_doc_counts AS
SELECT law, article, COUNT(*) AS cnt
FROM document_legislation_citations
GROUP BY law, article
"""

UPDATE_SQL = """
UPDATE legislation_embeddings e
SET doc_count = c.cnt
FROM tmp_doc_counts c
WHERE e.id >= $1 AND e.id < $2
  AND e.law = c.law
  AND COALESCE(e.article, '') = c.article
  AND e.doc_count IS DISTINCT FROM c.cnt
"""


async def main() -> None:
    conn = await asyncpg.connect(
        host=os.getenv('SUPABASE_DB_HOST'),
        port=int(os.getenv('SUPABASE_DB_PORT', '5432')),
        user=os.getenv('SUPABASE_DB_USER'),
        password=os.getenv('SUPABASE_DB_PASSWORD'),
        database=os.getenv('SUPABASE_DB_NAME', 'postgres'),
        statement_cache_size=0,
        command_timeout=1800,
    )
    await conn.execute("SET statement_timeout = '1800s'")

    before = await conn.fetchval(
        "SELECT COUNT(*) FROM legislation_embeddings WHERE COALESCE(doc_count, 0) = 0"
    )
    print(f"Rows with doc_count = 0 before: {before}", flush=True)

    t0 = time.time()
    await conn.execute(COUNTS_SQL)
    await conn.execute("CREATE INDEX ON tmp_doc_counts (law, article)")
    await conn.execute("ANALYZE tmp_doc_counts")
    print(f"Counted citations in {time.time() - t0:.0f}s", flush=True)

    max_id = await conn.fetchval("SELECT MAX(id) FROM legislation_embeddings")
    total = 0
    for start in range(0, max_id + 1, BATCH_IDS):
        status = await conn.execute(UPDATE_SQL, start, start + BATCH_IDS)
        total += int(status.split()[-1])
        print(f"  ids [{start}, {start + BATCH_IDS}): {status} — "
              f"{total} updated, {time.time() - t0:.0f}s elapsed", flush=True)

    after = await conn.fetchval(
        "SELECT COUNT(*) FROM legislation_embeddings WHERE COALESCE(doc_count, 0) = 0"
    )
    print(f"\nDone: {total} rows updated. Rows with doc_count = 0 after: {after}")

    rows = await conn.fetch(
        "SELECT law, article, doc_count FROM legislation_embeddings "
        "ORDER BY doc_count DESC NULLS LAST LIMIT 10"
    )
    print("\nTop 10 most-cited legislation after backfill:")
    for r in rows:
        print(f"  {r['law']} | art. {r['article']} | {r['doc_count']} docs")

    await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
