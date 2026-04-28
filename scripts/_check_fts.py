"""Inspect how the documents.fts column is built and which index serves it."""
import asyncio
import asyncpg


async def main():
    conn = await asyncpg.connect(
        host="aws-0-eu-west-1.pooler.supabase.com",
        port=5432,
        user="postgres.unorogsjlkimrehyndzs",
        password="*XH9B2RTSua23fR",
        database="postgres",
        statement_cache_size=0,
    )

    print("=== fts column definition ===")
    rows = await conn.fetch(
        """
        SELECT column_name, data_type, is_generated, generation_expression
          FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'fts'
        """
    )
    for r in rows:
        print(dict(r))

    print("\n=== fts index definition ===")
    rows = await conn.fetch(
        """
        SELECT indexname, indexdef
          FROM pg_indexes
         WHERE tablename = 'documents' AND indexname LIKE '%fts%'
        """
    )
    for r in rows:
        print(dict(r))

    print("\n=== sample fts value ===")
    row = await conn.fetchrow(
        "SELECT doc_id, summary, fts::text AS fts_text FROM documents WHERE fts IS NOT NULL LIMIT 1"
    )
    if row:
        print("doc_id:", row["doc_id"])
        print("summary:", (row["summary"] or "")[:120])
        print("fts (first 300 chars):", (row["fts_text"] or "")[:300])

    print("\n=== Try EXPLAIN ANALYZE with the actual API query (ORDER BY ts_rank_cd) ===")
    sql = (
        "SELECT doc_id, ts_rank_cd(fts, websearch_to_tsquery('portuguese', $1))::real AS rank "
        "  FROM documents "
        " WHERE fts @@ websearch_to_tsquery('portuguese', $1) "
        " ORDER BY rank DESC "
        " LIMIT 12"
    )
    plan = await conn.fetch("EXPLAIN ANALYZE " + sql, "responsabilidade civil")
    for p in plan:
        print("   ", p[0])

    print("\n=== Same query with enable_seqscan=off ===")
    await conn.execute("SET enable_seqscan = off")
    plan = await conn.fetch("EXPLAIN ANALYZE " + sql, "responsabilidade civil")
    for p in plan:
        print("   ", p[0])
    await conn.execute("RESET enable_seqscan")

    await conn.close()


asyncio.run(main())
