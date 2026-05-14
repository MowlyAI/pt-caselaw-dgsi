import asyncio, asyncpg, os
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
    total = await conn.fetchval('SELECT COUNT(*) FROM documents')
    print(f'Total documents: {total}')

    rows = await conn.fetch(
        """SELECT metadata->'legislation_cited' as lc
        FROM documents
        WHERE metadata->'legislation_cited' IS NOT NULL
        LIMIT 3"""
    )
    for r in rows:
        print('Sample:', r['lc'][:500] if isinstance(r['lc'], str) else r['lc'])

    print('\nBuilding temp table of all unique legislation pairs (this may take a while)...')
    await conn.execute("SET statement_timeout = '300000'")
    await conn.execute("""
        CREATE TEMP TABLE IF NOT EXISTS tmp_leg_uniq AS
        SELECT DISTINCT l->>'law' as law, l->>'article' as article
        FROM documents, jsonb_array_elements(metadata->'legislation_cited') AS l
    """)

    sample = await conn.fetch("SELECT law, article FROM tmp_leg_uniq LIMIT 10")
    print('\nSample unique legislation pairs:')
    for row in sample:
        print(f"  {row['law']} | {row['article']}")

    cnt = await conn.fetchval("SELECT COUNT(*) FROM tmp_leg_uniq")
    print(f'\nTotal unique (law, article) pairs: {cnt}')

    total_entries = await conn.fetchval(
        """SELECT COUNT(*) FROM documents, jsonb_array_elements(metadata->'legislation_cited') AS l"""
    )
    print(f'Total legislation entries (with duplicates): {total_entries}')

    top_laws = await conn.fetch(
        """SELECT l->>'law' as law, COUNT(*) as c
        FROM documents, jsonb_array_elements(metadata->'legislation_cited') AS l
        GROUP BY l->>'law'
        ORDER BY c DESC
        LIMIT 20"""
    )
    print('\nTop 20 laws by citation count:')
    for row in top_laws:
        print(f"  {row['law']}: {row['c']}")

    await conn.execute("DROP TABLE IF EXISTS tmp_leg_uniq")
    await conn.close()

if __name__ == '__main__':
    asyncio.run(main())
