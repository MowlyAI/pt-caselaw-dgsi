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
    rows = await conn.fetch("""
        SELECT pid, state, query_start, now() - query_start AS duration, query
        FROM pg_stat_activity
        WHERE state = 'active' AND query NOT LIKE '%pg_stat_activity%'
        ORDER BY duration DESC
    """)
    for r in rows:
        print(f"PID {r['pid']} | {r['state']} | {r['duration']} | {r['query'][:200]}")
    await conn.close()

asyncio.run(main())
