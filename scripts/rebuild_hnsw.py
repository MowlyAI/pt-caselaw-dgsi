"""Drop invalid HNSW indexes and rebuild them on Supabase.

Run this AFTER scaling compute up (Large or XL recommended).
Monitors progress via pg_stat_progress_create_index.
"""
import asyncio
import time

import asyncpg

PW = "*XH9B2RTSua23fR"
HOST = "aws-0-eu-west-1.pooler.supabase.com"
USER = "postgres.unorogsjlkimrehyndzs"

INDEXES = [
    ("idx_documents_embedding", "embedding"),
    ("idx_documents_embedding_context", "embedding_context"),
    ("idx_documents_embedding_ratio", "embedding_ratio"),
]


async def wait_for_index_build(conn):
    while True:
        rows = await conn.fetch("""
            SELECT now() - a.xact_start AS duration,
                   p.phase,
                   p.blocks_done,
                   p.blocks_total,
                   p.tuples_done,
                   p.tuples_total
              FROM pg_stat_progress_create_index p
              JOIN pg_stat_activity a ON p.pid = a.pid
        """)
        if not rows:
            print("  No active index build.")
            break
        for r in rows:
            dur = str(r["duration"]) if r["duration"] else "?"
            print(
                f"  [{dur}] phase={r['phase']} "
                f"blocks={r['blocks_done']}/{r['blocks_total']} "
                f"tuples={r['tuples_done']}/{r['tuples_total']}"
            )
        await asyncio.sleep(10)


async def main():
    conn = await asyncpg.connect(
        host=HOST,
        port=5432,
        user=USER,
        password=PW,
        database="postgres",
        statement_cache_size=0,
    )

    # 1. Check current state
    print("Current HNSW indexes:")
    rows = await conn.fetch("""
        SELECT indexrelid::regclass AS name,
               indisvalid,
               indisready,
               pg_size_pretty(pg_relation_size(indexrelid)) AS size
          FROM pg_index
         WHERE indrelid = 'documents'::regclass
           AND indexrelid::regclass::text LIKE 'idx_documents_embedding%'
         ORDER BY name
    """)
    for r in rows:
        print(f"  {r['name']}: valid={r['indisvalid']}, ready={r['indisready']}, size={r['size']}")

    # 2. Drop invalid indexes
    print("\nDropping invalid indexes...")
    for idx_name, _ in INDEXES:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_indexes WHERE indexname = $1", idx_name
        )
        if exists:
            await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {idx_name}")
            print(f"  Dropped {idx_name}")
        else:
            print(f"  {idx_name} already gone")

    # 3. Build new indexes
    print("\nBuilding new HNSW indexes...")
    for idx_name, col in INDEXES:
        print(f"\n  Creating {idx_name} on {col}...")
        t0 = time.time()
        # CONCURRENTLY prevents table lock but takes longer
        await conn.execute(f"""
            CREATE INDEX CONCURRENTLY {idx_name}
            ON documents USING hnsw ({col} halfvec_cosine_ops)
            WITH (m = 16, ef_construction = 200)
        """)
        print(f"  Started {idx_name}. Monitoring...")
        await wait_for_index_build(conn)
        print(f"  Done in {time.time()-t0:.1f}s")

    # 4. Verify
    print("\nVerifying new indexes:")
    rows = await conn.fetch("""
        SELECT indexrelid::regclass AS name,
               indisvalid,
               indisready,
               pg_size_pretty(pg_relation_size(indexrelid)) AS size
          FROM pg_index
         WHERE indrelid = 'documents'::regclass
           AND indexrelid::regclass::text LIKE 'idx_documents_embedding%'
         ORDER BY name
    """)
    for r in rows:
        print(f"  {r['name']}: valid={r['indisvalid']}, ready={r['indisready']}, size={r['size']}")
        if not r["indisvalid"]:
            print("  WARNING: index is still invalid!")

    # 5. Quick test
    print("\nTesting vector search...")
    emb = await conn.fetchval(
        "SELECT embedding::text FROM documents WHERE embedding IS NOT NULL LIMIT 1"
    )
    await conn.execute("SET hnsw.ef_search = 40")
    t0 = time.time()
    rows = await conn.fetch("""
        SELECT doc_id,
               (1 - (embedding <=> $1::halfvec))::real AS sim
          FROM documents
         WHERE embedding IS NOT NULL
         ORDER BY embedding <=> $1::halfvec
         LIMIT 5
    """, emb)
    print(f"  Query time: {time.time()-t0:.2f}s")
    for r in rows:
        print(f"    {r['doc_id']} sim={r['sim']:.4f}")

    await conn.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
