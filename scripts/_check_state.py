import asyncio, asyncpg

pw = '*XH9B2RTSua23fR'

async def main():
    conn = await asyncpg.connect(
        host='aws-0-eu-west-1.pooler.supabase.com',
        port=5432,
        user='postgres.unorogsjlkimrehyndzs',
        password=pw,
        database='postgres',
        statement_cache_size=0,
    )
    print('=== HNSW indexes ===')
    rows = await conn.fetch('''
        SELECT indexrelid::regclass AS name, indisvalid, indisready,
               pg_size_pretty(pg_relation_size(indexrelid)) AS size
          FROM pg_index
         WHERE indrelid = 'documents'::regclass
           AND indexrelid::regclass::text LIKE 'idx_documents_embedding%'
         ORDER BY name
    ''')
    for r in rows:
        print(f"  {r['name']}: valid={r['indisvalid']}, ready={r['indisready']}, size={r['size']}")
    
    print('\n=== Instance config ===')
    for name in ['shared_buffers', 'effective_cache_size', 'random_page_cost', 'max_connections']:
        val = await conn.fetchval("SELECT setting FROM pg_settings WHERE name = $1", name)
        print(f"  {name} = {val}")
    
    print('\n=== Active index builds ===')
    rows = await conn.fetch('''
        SELECT p.pid, now() - a.xact_start AS duration, p.phase,
               p.blocks_done, p.blocks_total, p.tuples_done, p.tuples_total
          FROM pg_stat_progress_create_index p
          JOIN pg_stat_activity a ON p.pid = a.pid
    ''')
    print(f"  {len(rows)} active builds")
    for r in rows:
        print(f"    {dict(r)}")
    
    await conn.close()

asyncio.run(main())
