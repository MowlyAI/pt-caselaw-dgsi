"""Test that vector search now uses the new HNSW index and is fast."""
import asyncio
import time

import asyncpg

PW = "*XH9B2RTSua23fR"
HOST = "aws-0-eu-west-1.pooler.supabase.com"
USER = "postgres.unorogsjlkimrehyndzs"


async def main():
    conn = await asyncpg.connect(
        host=HOST, port=5432, user=USER, password=PW,
        database="postgres", statement_cache_size=0,
        command_timeout=120,
    )

    # Get a sample embedding from an existing row
    emb = await conn.fetchval(
        "SELECT embedding::text FROM documents WHERE embedding IS NOT NULL LIMIT 1"
    )
    print(f"Using sample embedding (first 60 chars): {emb[:60]}...")

    # Set ef_search for HNSW
    await conn.execute("SET hnsw.ef_search = 40")

    # Time vector search
    print("\n=== Vector search (HNSW) ===")
    t0 = time.time()
    rows = await conn.fetch("""
        SELECT doc_id, (1 - (embedding <=> $1::halfvec))::real AS sim
          FROM documents
         WHERE embedding IS NOT NULL
         ORDER BY embedding <=> $1::halfvec
         LIMIT 5
    """, emb)
    elapsed = time.time() - t0
    print(f"Query time: {elapsed*1000:.0f} ms")
    for r in rows:
        print(f"  {r['doc_id']} sim={r['sim']:.4f}")

    # Run again to see warm cache
    print("\n=== Vector search (HNSW, warm) ===")
    t0 = time.time()
    rows = await conn.fetch("""
        SELECT doc_id, (1 - (embedding <=> $1::halfvec))::real AS sim
          FROM documents
         WHERE embedding IS NOT NULL
         ORDER BY embedding <=> $1::halfvec
         LIMIT 5
    """, emb)
    elapsed = time.time() - t0
    print(f"Query time: {elapsed*1000:.0f} ms")

    # EXPLAIN ANALYZE to confirm index usage
    print("\n=== EXPLAIN ANALYZE ===")
    rows = await conn.fetch("""
        EXPLAIN ANALYZE
        SELECT doc_id, (1 - (embedding <=> $1::halfvec))::real AS sim
          FROM documents
         WHERE embedding IS NOT NULL
         ORDER BY embedding <=> $1::halfvec
         LIMIT 5
    """, emb)
    for r in rows:
        print(f"  {r[0]}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
