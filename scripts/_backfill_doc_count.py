import asyncio
import os

import asyncpg
from dotenv import load_dotenv

load_dotenv('.env.local')


async def main():
    conn = await asyncpg.connect(
        host=os.getenv('SUPABASE_DB_HOST'),
        port=int(os.getenv('SUPABASE_DB_PORT', '5432')),
        user=os.getenv('SUPABASE_DB_USER'),
        password=os.getenv('SUPABASE_DB_PASSWORD'),
        database=os.getenv('SUPABASE_DB_NAME', 'postgres'),
    )

    print('Backfilling doc_count from documents table...')
    await conn.execute("SET statement_timeout = '0'")  # disable timeout for this heavy operation

    # Step 1: build a temp table with counts (same trick as build script)
    print('  Creating temp table with citation counts...')
    await conn.execute("DROP TABLE IF EXISTS tmp_leg_counts")
    await conn.execute("""
        CREATE TEMP TABLE tmp_leg_counts AS
        SELECT l->>'law' as law, l->>'article' as article, COUNT(*) as cnt
        FROM documents, jsonb_array_elements(metadata->'legislation_cited') AS l
        WHERE l->>'law' IS NOT NULL AND l->>'law' != ''
        GROUP BY l->>'law', l->>'article'
    """)
    await conn.execute("CREATE INDEX ON tmp_leg_counts (law, article)")
    total = await conn.fetchval("SELECT COUNT(*) FROM tmp_leg_counts")
    print(f'  Temp table ready: {total} unique (law, article) pairs')

    # Step 2: update in chunks to avoid long transaction hold
    print('  Updating legislation_embeddings in chunks...')
    offset = 0
    batch_size = 50000
    total_updated = 0
    while True:
        result = await conn.execute(f"""
            UPDATE legislation_embeddings e
            SET doc_count = t.cnt
            FROM tmp_leg_counts t
            WHERE e.law = t.law AND (e.article IS NOT DISTINCT FROM t.article)
            AND e.id IN (
                SELECT id FROM legislation_embeddings
                ORDER BY law, article
                LIMIT {batch_size} OFFSET {offset}
            )
        """)
        # asyncpg execute returns the command tag string like "UPDATE 12345"
        count = int(result.split()[1]) if result.split() else 0
        total_updated += count
        if count == 0:
            break
        offset += batch_size
        print(f'    Updated {total_updated} rows so far...')

    print(f'  Total rows updated: {total_updated}')

    print(f'Update result: {result}')

    # Show top 5 after update
    rows = await conn.fetch(
        'SELECT law, article, citation_text, doc_count FROM legislation_embeddings ORDER BY doc_count DESC NULLS LAST LIMIT 5'
    )
    for r in rows:
        print(f'  {r["law"]} | art. {r["article"]} | cited {r["doc_count"]} times')

    await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
