"""Populate document_judges from the page's "Relator:" field.

Reads the relator with api.judges (page field first, metadata.judge_name
otherwise), stores it with its normalised judge_key, and refreshes the
judge_directory summary used by GET /judges. Decisions with no relator lose
their row. Runs one court at a time so each statement stays bounded.

Usage:
    python scripts/backfill_judges.py                           # every document
    python scripts/backfill_judges.py --updated-since 2026-10-08 # incremental
    python scripts/backfill_judges.py --dry-run
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
from api.judges import judge_key, judge_name  # noqa: E402

load_dotenv(".env.local")

# Only the text after the label leaves the database, not the whole decision.
RELATOR_SQL = """
SELECT doc_id,
       CASE WHEN strpos(full_text, 'Relator:') > 0
            THEN substr(full_text, strpos(full_text, 'Relator:'), 300) END AS relator_text,
       metadata->>'judge_name' AS extracted
FROM documents
WHERE court_short = $1
  {since_filter}
"""
SINCE_FILTER = "AND updated_at >= $2"
UPSERT_SQL = """
INSERT INTO document_judges (doc_id, judge_name, judge_key)
SELECT * FROM unnest($1::text[], $2::text[], $3::text[])
ON CONFLICT (doc_id) DO UPDATE
SET judge_name = EXCLUDED.judge_name, judge_key = EXCLUDED.judge_key
"""
DELETE_SQL = "DELETE FROM document_judges WHERE doc_id = ANY($1::text[])"


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
    total = from_page = 0
    for court in courts:
        rows = await conn.fetch(RELATOR_SQL.format(since_filter=since_filter), court, *since_args)
        keep: dict[str, str] = {}
        drop: list[str] = []
        for row in rows:
            name = judge_name(row["relator_text"], {"judge_name": row["extracted"]})
            if name:
                keep[row["doc_id"]] = name
                from_page += judge_name(row["relator_text"], None) is not None
            else:
                drop.append(row["doc_id"])
        total += len(keep)
        print(f"  {court}: {len(rows)} documents, {len(keep)} with a relator "
              f"({time.time() - t0:.0f}s)", flush=True)
        if dry_run:
            sample = list(keep.values())[:5]
            print(f"    e.g. {json.dumps(sample, ensure_ascii=False)}", flush=True)
            continue
        async with conn.transaction():
            if keep:
                names = list(keep.values())
                await conn.execute(UPSERT_SQL, list(keep), names, [judge_key(n) for n in names])
            if drop:
                await conn.execute(DELETE_SQL, drop)
    if not dry_run:
        await conn.execute("ANALYZE document_judges")
        await conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY judge_directory")
    print(f"Done. {total} documents with a relator ({from_page} from the page field)"
          f"{' (dry run, nothing written)' if dry_run else ''}.")
    await conn.close()


@click.command()
@click.option("--updated-since", type=click.DateTime(), default=None,
              help="Only re-read documents written since this date")
@click.option("--dry-run", is_flag=True, help="Read and count without writing")
def main(updated_since: Optional[datetime], dry_run: bool) -> None:
    asyncio.run(run(updated_since, dry_run))


if __name__ == "__main__":
    main()
