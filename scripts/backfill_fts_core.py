"""Populate document_fts_core for documents that have no row yet.

New and updated documents are kept in sync by the documents_fts_core_sync
trigger (migrations/20261010_search_quality_fts_core_dedup_legal_areas.sql);
this script backfills existing rows and repairs any gaps. It runs one court
at a time so each statement stays bounded.

Usage:
    python scripts/backfill_fts_core.py            # only missing rows
    python scripts/backfill_fts_core.py --rebuild  # recompute every row
"""
import asyncio
import os
import time

import asyncpg
import click
from dotenv import load_dotenv

load_dotenv(".env.local")

FILL_SQL = """
INSERT INTO document_fts_core (doc_id, fts_core)
SELECT d.doc_id, document_fts_core_vector(d.summary, d.metadata)
FROM documents d
WHERE d.court_short = $1
  {missing_only}
ON CONFLICT (doc_id) DO UPDATE SET fts_core = EXCLUDED.fts_core
"""
MISSING_ONLY = "AND NOT EXISTS (SELECT 1 FROM document_fts_core c WHERE c.doc_id = d.doc_id)"


async def run(rebuild: bool) -> None:
    conn = await asyncpg.connect(
        host=os.getenv("SUPABASE_DB_HOST"),
        port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
        user=os.getenv("SUPABASE_DB_USER"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        database=os.getenv("SUPABASE_DB_NAME", "postgres"),
        statement_cache_size=0,
        command_timeout=3600,
    )
    await conn.execute("SET statement_timeout = '3600s'")
    sql = FILL_SQL.format(missing_only="" if rebuild else MISSING_ONLY)
    courts = [r["court_short"] for r in await conn.fetch(
        "SELECT court_short, count(*) FROM documents GROUP BY 1 ORDER BY 2"
    )]
    t0 = time.time()
    for court in courts:
        status = await conn.execute(sql, court)
        print(f"  {court}: {status} ({time.time() - t0:.0f}s)", flush=True)
    await conn.execute("ANALYZE document_fts_core")
    missing = await conn.fetchval(
        "SELECT count(*) FROM documents d WHERE NOT EXISTS "
        "(SELECT 1 FROM document_fts_core c WHERE c.doc_id = d.doc_id)"
    )
    print(f"Done. Documents still without fts_core: {missing}")
    await conn.close()


@click.command()
@click.option("--rebuild", is_flag=True, help="Recompute rows that already exist")
def main(rebuild: bool) -> None:
    asyncio.run(run(rebuild))


if __name__ == "__main__":
    main()
