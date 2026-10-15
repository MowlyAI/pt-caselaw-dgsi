"""Populate document_disability_points from the extracted injuries.

Parses metadata.injuries[].disability_degree with api.disability and stores
the decision's highest permanent disability degree. Decisions whose degree no
longer parses lose their row. Runs one court at a time so each statement stays
bounded.

Usage:
    python scripts/backfill_disability_points.py                          # every document
    python scripts/backfill_disability_points.py --updated-since 2026-10-08 # incremental
    python scripts/backfill_disability_points.py --dry-run
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncpg
import click
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.disability import disability_points  # noqa: E402

load_dotenv(".env.local")

INJURIES_SQL = """
SELECT doc_id, metadata->'injuries' AS injuries
FROM documents
WHERE court_short = $1
  AND jsonb_typeof(metadata->'injuries') = 'array'
  AND jsonb_array_length(metadata->'injuries') > 0
  {since_filter}
"""
SINCE_FILTER = "AND updated_at >= $2"
UPSERT_SQL = """
INSERT INTO document_disability_points (doc_id, points)
SELECT * FROM unnest($1::text[], $2::real[])
ON CONFLICT (doc_id) DO UPDATE SET points = EXCLUDED.points
"""
DELETE_SQL = """
DELETE FROM document_disability_points
WHERE doc_id = ANY($1::text[])
"""
STALE_SQL = """
DELETE FROM document_disability_points p
USING documents d
WHERE d.doc_id = p.doc_id AND d.court_short = $1
  AND NOT (d.doc_id = ANY($2::text[]))
  {since_filter}
"""
STALE_SINCE_FILTER = "AND d.updated_at >= $3"


async def run(updated_since: Optional[datetime], dry_run: bool) -> None:
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
    since_filter = SINCE_FILTER if updated_since else ""
    since_args = [updated_since] if updated_since else []
    courts = [r["court_short"] for r in await conn.fetch(
        "SELECT court_short, count(*) FROM documents GROUP BY 1 ORDER BY 2"
    )]
    t0 = time.time()
    total = 0
    for court in courts:
        rows = await conn.fetch(INJURIES_SQL.format(since_filter=since_filter), court, *since_args)
        parsed = {}
        for row in rows:
            injuries = row["injuries"]
            if isinstance(injuries, str):
                injuries = json.loads(injuries)
            parsed[row["doc_id"]] = disability_points({"injuries": injuries})
        keep = {doc_id: points for doc_id, points in parsed.items() if points is not None}
        drop = [doc_id for doc_id, points in parsed.items() if points is None]
        total += len(keep)
        print(f"  {court}: {len(rows)} with injuries, {len(keep)} with points "
              f"({time.time() - t0:.0f}s)", flush=True)
        if dry_run:
            continue
        async with conn.transaction():
            if keep:
                await conn.execute(UPSERT_SQL, list(keep), list(keep.values()))
            if drop:
                await conn.execute(DELETE_SQL, drop)
            # Rows of documents that no longer list any injury.
            await conn.execute(
                STALE_SQL.format(since_filter=STALE_SINCE_FILTER if updated_since else ""),
                court, list(parsed), *since_args,
            )
    if not dry_run:
        await conn.execute("ANALYZE document_disability_points")
    print(f"Done. {total} documents with a disability degree"
          f"{' (dry run, nothing written)' if dry_run else ''}.")
    await conn.close()


@click.command()
@click.option("--updated-since", type=click.DateTime(), default=None,
              help="Only re-parse documents written since this date")
@click.option("--dry-run", is_flag=True, help="Parse and count without writing")
def main(updated_since: Optional[datetime], dry_run: bool) -> None:
    asyncio.run(run(updated_since, dry_run))


if __name__ == "__main__":
    main()
