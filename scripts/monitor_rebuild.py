"""Monitor and finish HNSW index rebuild on Supabase."""
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


async def monitor_build(conn, idx_name):
    """Wait for active index build to finish."""
    print(f"  Monitoring {idx_name}...")
    while True:
        rows = await conn.fetch("""
            SELECT now() - a.xact_start AS duration,
                   p.phase, p.blocks_done, p.blocks_total,
                   p.tuples_done, p.tuples_total
              FROM pg_stat_progress_create_index p
              JOIN pg_stat_activity a ON p.pid = a.pid
        """)
        if not rows:
            print(f"  {idx_name} build complete!")
            return
        for r in rows:
            dur = str(r["duration"]) if r["duration"] else "?"
            pct = (r["blocks_done"] / max(r["blocks_total"], 1) * 100) if r["blocks_total"] else 0
            print(
                f"  [{dur}] phase={r['phase']} "
                f"blocks={r['blocks_done']}/{r['blocks_total']} ({pct:.1f}%) "
                f"tuples={r['tuples_done']}"
            )
        await asyncio.sleep(15)


async def build_one(conn, idx_name, col_name):
    """Create an index with monitoring."""
    # Drop any existing (partial or invalid)
    exists = await conn.fetchval(
        "SELECT 1 FROM pg_indexes WHERE indexname = $1", idx_name
    )
    if exists:
        print(f"  Dropping existing {idx_name}...")
        await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {idx_name}")

    print(f"  Creating {idx_name} on {col_name}...")
    t0 = time.time()
    # Fire and forget - run in background
    await conn.execute(f"""
        CREATE INDEX CONCURRENTLY {idx_name}
        ON documents USING hnsw ({col_name} halfvec_cosine_ops)
        WITH (m = 16, ef_construction = 200)
    """)
    print(f"  {idx_name} created in {time.time()-t0:.1f}s")


async def main():
    conn = await asyncpg.connect(
        host=HOST, port=5432, user=USER, password=PW,
        database="postgres", statement_cache_size=0,
        command_timeout=3600,  # 1 hour per command
    )

    # Check current state
    print("Checking indexes...")
    rows = await conn.fetch("""
        SELECT indexrelid::regclass AS name, indisvalid, indisready,
               pg_size_pretty(pg_relation_size(indexrelid)) AS size
          FROM pg_index
         WHERE indrelid = 'documents'::regclass
           AND indexrelid::regclass::text LIKE 'idx_documents_embedding%'
         ORDER BY name
    """)
    current = {r["name"]: r for r in rows}
    for name, info in current.items():
        print(f"  {name}: valid={info['indisvalid']}, ready={info['indisready']}, size={info['size']}")

    # Check if a build is already in progress
    progress = await conn.fetch("""
        SELECT p.phase, now() - a.xact_start AS duration
          FROM pg_stat_progress_create_index p
          JOIN pg_stat_activity a ON p.pid = a.pid
    """)
    if progress:
        print(f"\nActive build detected: {progress[0]['phase']} (running for {progress[0]['duration']})")
        await monitor_build(conn, "in-progress index")
        print("\nRe-checking indexes after build...")
        rows = await conn.fetch("""
            SELECT indexrelid::regclass AS name, indisvalid, indisready,
                   pg_size_pretty(pg_relation_size(indexrelid)) AS size
              FROM pg_index
             WHERE indrelid = 'documents'::regclass
               AND indexrelid::regclass::text LIKE 'idx_documents_embedding%'
             ORDER BY name
        """)
        current = {r["name"]: r for r in rows}

    # Build remaining indexes
    for idx_name, col_name in INDEXES:
        info = current.get(idx_name)
        if info and info["indisvalid"]:
            print(f"\n{idx_name} is already valid ({info['size']}) — skipping.")
            continue

        await build_one(conn, idx_name, col_name)
        await monitor_build(conn, idx_name)

    # Final verification
    print("\n=== Final verification ===")
    for idx_name, col_name in INDEXES:
        info = await conn.fetchrow("""
            SELECT indexrelid::regclass AS name, indisvalid, indisready,
                   pg_size_pretty(pg_relation_size(indexrelid)) AS size
              FROM pg_index
             WHERE indrelid = 'documents'::regclass
               AND indexrelid::regclass::text = $1
        """, idx_name)
        if info:
            print(f"  {info['name']}: valid={info['indisvalid']}, ready={info['indisready']}, size={info['size']}")
        else:
            print(f"  {idx_name}: MISSING!")

    # Quick vector search test
    print("\n=== Vector search test ===")
    emb = await conn.fetchval(
        "SELECT embedding::text FROM documents WHERE embedding IS NOT NULL LIMIT 1"
    )
    await conn.execute("SET hnsw.ef_search = 40")
    t0 = time.time()
    rows = await conn.fetch("""
        SELECT doc_id, (1 - (embedding <=> $1::halfvec))::real AS sim
          FROM documents
         WHERE embedding IS NOT NULL
         ORDER BY embedding <=> $1::halfvec
         LIMIT 5
    """, emb)
    print(f"Query time: {time.time()-t0:.2f}s")
    for r in rows:
        print(f"  {r['doc_id']} sim={r['sim']:.4f}")

    await conn.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
