"""Validate the candidate-cap FTS strategy."""
import asyncio
import time
import asyncpg


SQL = """
WITH cands AS (
  SELECT doc_id, fts
    FROM documents
   WHERE fts @@ websearch_to_tsquery('portuguese', $1)
   LIMIT $3
)
SELECT doc_id,
       ts_rank_cd(fts, websearch_to_tsquery('portuguese', $1))::real AS rank
  FROM cands
 ORDER BY rank DESC
 LIMIT $2
"""


async def main():
    conn = await asyncpg.connect(
        host="aws-0-eu-west-1.pooler.supabase.com",
        port=5432,
        user="postgres.unorogsjlkimrehyndzs",
        password="*XH9B2RTSua23fR",
        database="postgres",
        statement_cache_size=0,
    )

    queries = [
        "responsabilidade civil",
        "despedimento sem justa causa",
        "acidente de trabalho",
        "competencia material tribunal trabalho",
        "abuso direito locador arrendamento",
    ]

    for q in queries:
        for run in ("cold", "warm"):
            t0 = time.perf_counter()
            rows = await conn.fetch(SQL, q, 12, 1500)
            dt = (time.perf_counter() - t0) * 1000
            print(f"  {run:5s}  {dt:7.0f} ms   q={q!r}  hits={len(rows)}  top_rank={rows[0]['rank']:.4f}" if rows else f"  {run:5s}  {dt:7.0f} ms   q={q!r}  no hits")

    print("\n=== EXPLAIN ANALYZE for the broad query ===")
    plan = await conn.fetch("EXPLAIN ANALYZE " + SQL, "responsabilidade civil", 12, 1500)
    for p in plan:
        print("   ", p[0])

    await conn.close()


asyncio.run(main())
