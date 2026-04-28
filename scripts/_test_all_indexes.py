"""Verify all 3 HNSW indexes are used and fast."""
import asyncio
import time

import asyncpg

PW = "*XH9B2RTSua23fR"
HOST = "aws-0-eu-west-1.pooler.supabase.com"
USER = "postgres.unorogsjlkimrehyndzs"

COLUMNS = ["embedding", "embedding_context", "embedding_ratio"]


async def main():
    conn = await asyncpg.connect(
        host=HOST, port=5432, user=USER, password=PW,
        database="postgres", statement_cache_size=0,
        command_timeout=60,
    )

    await conn.execute("SET hnsw.ef_search = 40")

    for col in COLUMNS:
        print(f"\n=== {col} ===")
        emb = await conn.fetchval(
            f"SELECT {col}::text FROM documents WHERE {col} IS NOT NULL LIMIT 1"
        )

        # Cold + warm
        for label in ["cold", "warm"]:
            t0 = time.time()
            rows = await conn.fetch(
                f"SELECT doc_id, (1 - ({col} <=> $1::halfvec))::real AS sim "
                f"FROM documents WHERE {col} IS NOT NULL "
                f"ORDER BY {col} <=> $1::halfvec LIMIT 5",
                emb,
            )
            ms = (time.time() - t0) * 1000
            print(f"  {label}: {ms:.0f} ms (top sim={rows[0]['sim']:.4f})")

        # Confirm index usage
        plan = await conn.fetch(
            f"EXPLAIN SELECT doc_id FROM documents WHERE {col} IS NOT NULL "
            f"ORDER BY {col} <=> $1::halfvec LIMIT 5",
            emb,
        )
        for r in plan[:3]:
            print(f"    {r[0]}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
