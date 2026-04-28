import asyncio, asyncpg, time

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
    await conn.execute('SET hnsw.ef_search = 40')

    emb = '[' + ','.join(['0.01']*1024) + ']'
    t0 = time.time()
    rows = await conn.fetch('''
        SELECT doc_id, (1 - (embedding <=> $1::halfvec))::real AS sim
        FROM documents
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1::halfvec
        LIMIT 5
    ''', emb)
    print(f'HNSW query time: {time.time()-t0:.2f}s')
    for r in rows:
        print(r['doc_id'], r['sim'])

    t0 = time.time()
    rows = await conn.fetch('''
        SELECT doc_id, ts_rank_cd(fts, websearch_to_tsquery('portuguese', unaccent($1))) AS rank
        FROM documents
        WHERE fts @@ websearch_to_tsquery('portuguese', unaccent($1))
        ORDER BY rank DESC
        LIMIT 5
    ''', 'divorcio litigioso')
    print(f'FTS query time: {time.time()-t0:.2f}s')
    for r in rows:
        print(r['doc_id'], r['rank'])

    await conn.close()

asyncio.run(main())
