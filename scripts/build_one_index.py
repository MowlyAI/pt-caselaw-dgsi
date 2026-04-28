"""Build a single HNSW index. Pass the index name as argv[1].

Sets maintenance_work_mem high for the session so the HNSW graph fits in
memory during the build (this is the key difference vs the previous attempts).
"""
import asyncio
import sys
import time

import asyncpg

PW = "*XH9B2RTSua23fR"
HOST = "aws-0-eu-west-1.pooler.supabase.com"
USER = "postgres.unorogsjlkimrehyndzs"

INDEXES = {
    "idx_documents_embedding": "embedding",
    "idx_documents_embedding_context": "embedding_context",
    "idx_documents_embedding_ratio": "embedding_ratio",
}


async def main(idx_name: str):
    if idx_name not in INDEXES:
        print(f"Unknown index: {idx_name}. Choose from {list(INDEXES)}")
        sys.exit(1)
    col = INDEXES[idx_name]

    conn = await asyncpg.connect(
        host=HOST, port=5432, user=USER, password=PW,
        database="postgres", statement_cache_size=0,
        command_timeout=None,  # no timeout
    )

    # 1. Drop existing (valid or invalid) version of this index
    exists = await conn.fetchval(
        "SELECT 1 FROM pg_indexes WHERE indexname = $1", idx_name
    )
    if exists:
        print(f"Dropping existing {idx_name}...", flush=True)
        # Use plain DROP (not CONCURRENTLY) since the index is invalid/unused
        await conn.execute(f"DROP INDEX IF EXISTS {idx_name}")
        print("  Dropped.", flush=True)

    # 2. Tune session for the build
    print("Setting session GUCs for the build...", flush=True)
    await conn.execute("SET maintenance_work_mem = '1536MB'")
    # Parallel workers exhaust /dev/shm on Supabase Medium; force serial build.
    await conn.execute("SET max_parallel_maintenance_workers = 0")
    # Disable server-side timeouts so a long-running CREATE INDEX is not killed.
    await conn.execute("SET statement_timeout = 0")
    await conn.execute("SET idle_in_transaction_session_timeout = 0")
    await conn.execute("SET lock_timeout = 0")
    mwm = await conn.fetchval("SHOW maintenance_work_mem")
    workers = await conn.fetchval("SHOW max_parallel_maintenance_workers")
    stmt_to = await conn.fetchval("SHOW statement_timeout")
    print(f"  maintenance_work_mem = {mwm}", flush=True)
    print(f"  max_parallel_maintenance_workers = {workers}", flush=True)
    print(f"  statement_timeout = {stmt_to}", flush=True)

    # 3. Build the index. NOT CONCURRENTLY so it actually finishes in one go
    # (the table is read-only during the build, but that is acceptable).
    print(f"\nCreating {idx_name} on {col}...", flush=True)
    t0 = time.time()
    sql = (
        f"CREATE INDEX {idx_name} "
        f"ON documents USING hnsw ({col} halfvec_cosine_ops) "
        f"WITH (m = 16, ef_construction = 200)"
    )
    print(f"  SQL: {sql}", flush=True)
    await conn.execute(sql)
    elapsed = time.time() - t0
    print(f"\n  Built in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    # 4. Verify
    info = await conn.fetchrow("""
        SELECT indexrelid::regclass AS name, indisvalid, indisready,
               pg_size_pretty(pg_relation_size(indexrelid)) AS size
          FROM pg_index
         WHERE indrelid = 'documents'::regclass
           AND indexrelid::regclass::text = $1
    """, idx_name)
    print(
        f"  {info['name']}: valid={info['indisvalid']}, "
        f"ready={info['indisready']}, size={info['size']}",
        flush=True,
    )

    await conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: build_one_index.py <index_name>")
        print(f"Choices: {list(INDEXES)}")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
