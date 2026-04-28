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

    print('--- Indexes on documents ---')
    idx = await conn.fetch('''
        SELECT indexrelname, idx_scan, idx_tup_read, idx_tup_fetch
        FROM pg_stat_user_indexes
        WHERE schemaname = 'public' AND relname = 'documents'
        ORDER BY indexrelname
    ''')
    for r in idx:
        print(dict(r))

    print('\n--- Table size ---')
    sz = await conn.fetchval('''
        SELECT pg_size_pretty(pg_total_relation_size('documents'))
    ''')
    print(sz)

    print('\n--- Try with real embedding ---')
    # Use a real embedding from a random document
    real_emb = await conn.fetchval('''
        SELECT embedding::text FROM documents WHERE embedding IS NOT NULL LIMIT 1
    ''')
    print('emb len:', len(real_emb) if real_emb else 'none')

    if real_emb:
        t0 = __import__('time').time()
        rows = await conn.fetch('''
            SELECT doc_id, (1 - (embedding <=> $1::halfvec))::real AS sim
            FROM documents
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::halfvec
            LIMIT 5
        ''', real_emb)
        print(f'query time: {__import__("time").time()-t0:.2f}s')
        for r in rows:
            print(r['doc_id'], r['sim'])

    await conn.close()

asyncio.run(main())
