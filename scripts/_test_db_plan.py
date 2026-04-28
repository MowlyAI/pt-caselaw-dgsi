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
    emb = '[' + ','.join(['0.01']*1024) + ']'
    
    plan = await conn.fetch('''
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT doc_id, (1 - (embedding <=> $1::halfvec))::real AS sim
        FROM documents
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1::halfvec
        LIMIT 5
    ''', emb)
    for row in plan:
        print(row['QUERY PLAN'])

    print('\n--- Index usage ---')
    idx = await conn.fetch('''
        SELECT indexname, idx_scan, idx_tup_read, idx_tup_fetch
        FROM pg_stat_user_indexes
        WHERE schemaname = 'public' AND relname = 'documents'
        AND indexname LIKE 'idx_%'
        ORDER BY indexname
    ''')
    for r in idx:
        print(r)

    print('\n--- Table size ---')
    sz = await conn.fetchval('''
        SELECT pg_size_pretty(pg_total_relation_size('documents'))
    ''')
    print(sz)

    await conn.close()

asyncio.run(main())
