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
    mem = await conn.fetchval("SELECT setting FROM pg_settings WHERE name = 'shared_buffers'")
    max_conn = await conn.fetchval("SELECT setting FROM pg_settings WHERE name = 'max_connections'")
    cache = await conn.fetchval("SELECT setting FROM pg_settings WHERE name = 'effective_cache_size'")
    rpc = await conn.fetchval("SELECT setting FROM pg_settings WHERE name = 'random_page_cost'")
    print(f'shared_buffers={mem}, max_connections={max_conn}')
    print(f'effective_cache_size={cache}')
    print(f'random_page_cost={rpc}')
    await conn.close()

asyncio.run(main())
