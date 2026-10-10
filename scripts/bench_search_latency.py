"""Measure search latency with the same SQL the API runs (vector + FTS).

Query vectors are embeddings of random stored documents, so no embedding API
calls are needed; each run samples fresh ones so results are not cache-primed.

Usage: python scripts/bench_search_latency.py [--n 30]
"""
import asyncio
import os
import statistics
import time

import asyncpg
import click
from dotenv import load_dotenv

load_dotenv(".env.local")

FIELDS = ["embedding", "embedding_context", "embedding_ratio"]
FTS_QUERIES = [
    "impugnação da matéria de facto", "responsabilidade civil extracontratual",
    "despedimento ilícito", "insolvência culposa", "contrato de arrendamento",
    "acidente de viação", "nulidade da sentença", "prescrição", "usucapião",
    "abuso de direito", "erro na apreciação da prova", "indemnização por danos não patrimoniais",
    "execução fiscal", "contrato de empreitada", "caso julgado",
]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


async def run(n: int) -> None:
    conn = await asyncpg.connect(
        host=os.getenv("SUPABASE_DB_HOST"), port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
        user=os.getenv("SUPABASE_DB_USER"), password=os.getenv("SUPABASE_DB_PASSWORD"),
        database=os.getenv("SUPABASE_DB_NAME", "postgres"), statement_cache_size=0,
    )
    await conn.execute(f"SET hnsw.ef_search = {int(os.getenv('HNSW_EF_SEARCH', '40'))}")
    await conn.execute("SET hnsw.iterative_scan = strict_order")
    print("instance:", await conn.fetchval("SHOW shared_buffers"), "shared_buffers")
    rtt = []
    for _ in range(10):
        t = time.perf_counter(); await conn.fetchval("SELECT 1"); rtt.append((time.perf_counter() - t) * 1000)
    print(f"  network round trip (SELECT 1) p50={statistics.median(rtt):.0f} ms")
    vecs = [r["e"] for r in await conn.fetch(
        "SELECT embedding::text AS e FROM documents TABLESAMPLE SYSTEM (0.5) "
        "WHERE embedding IS NOT NULL LIMIT $1", n)]
    results = {}
    for field in FIELDS:
        times = []
        for v in vecs:
            t = time.perf_counter()
            await conn.fetch(
                f"SELECT doc_id, (1 - ({field} <=> $1::halfvec))::real AS sim FROM documents "
                f"WHERE {field} IS NOT NULL ORDER BY {field} <=> $1::halfvec LIMIT 40", v)
            times.append((time.perf_counter() - t) * 1000)
        results[f"vector {field}"] = times
    times = []
    for q in FTS_QUERIES * max(1, n // len(FTS_QUERIES)):
        t = time.perf_counter()
        await conn.fetch(
            "WITH cands AS (SELECT doc_id, fts FROM documents "
            "WHERE fts @@ websearch_to_tsquery('portuguese', $1) LIMIT 1500) "
            "SELECT doc_id, ts_rank_cd(fts, websearch_to_tsquery('portuguese', $1))::real AS rank "
            "FROM cands ORDER BY rank DESC LIMIT 40", q)
        times.append((time.perf_counter() - t) * 1000)
    results["full-text"] = times
    for name, ts in results.items():
        print(f"  {name:28} p50={statistics.median(ts):7.0f} ms  p95={pct(ts, 0.95):7.0f} ms  max={max(ts):7.0f} ms  (n={len(ts)})")
    await conn.close()


@click.command()
@click.option("--n", default=30, help="Queries per search type")
def main(n: int) -> None:
    asyncio.run(run(n))


if __name__ == "__main__":
    main()
