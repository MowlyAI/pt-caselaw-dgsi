"""Mark duplicate DGSI decisions so search returns each decision once.

DGSI sometimes publishes the same decision twice under different document
ids. Documents sharing court + process number + decision date are compared;
copies whose text is near-identical to the kept one get
`duplicate_of = <kept doc_id>` and are excluded by the API. Groups whose texts
differ are left alone (e.g. two distinct rulings in one process on one day).

The kept copy is the one with the longest full text (a full decision beats a
summary-only publication), then higher extraction confidence, then the
earliest created.

Usage:
    python scripts/mark_duplicates.py                            # whole corpus
    python scripts/mark_duplicates.py --updated-since 2026-10-08 # incremental
    python scripts/mark_duplicates.py --dry-run
"""
import asyncio
import difflib
import os
from datetime import datetime
from typing import Optional

import asyncpg
import click
from dotenv import load_dotenv

load_dotenv(".env.local")

SIMILARITY_THRESHOLD = 0.9
COMPARE_CHARS = 3000
CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}

GROUPS_SQL = """
SELECT court_short, process_number, decision_date, array_agg(doc_id) AS doc_ids
FROM documents
WHERE coalesce(process_number, '') <> ''
  {since_filter}
GROUP BY court_short, process_number, decision_date
HAVING count(*) > 1
"""
# Incremental mode: only groups that contain a recently written document.
SINCE_FILTER = """
  AND (court_short, process_number, decision_date) IN (
      SELECT court_short, process_number, decision_date FROM documents
      WHERE updated_at >= $1 AND coalesce(process_number, '') <> '')
"""


def _normalized(text: str) -> str:
    return " ".join((text or "").split())[:COMPARE_CHARS]


def _similar(a: str, b: str) -> bool:
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= SIMILARITY_THRESHOLD


async def run(updated_since: Optional[datetime], dry_run: bool) -> None:
    conn = await asyncpg.connect(
        host=os.getenv("SUPABASE_DB_HOST"),
        port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
        user=os.getenv("SUPABASE_DB_USER"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        database=os.getenv("SUPABASE_DB_NAME", "postgres"),
        statement_cache_size=0,
        command_timeout=1800,
    )
    await conn.execute("SET statement_timeout = '1800s'")
    if updated_since:
        groups = await conn.fetch(GROUPS_SQL.format(since_filter=SINCE_FILTER), updated_since)
    else:
        groups = await conn.fetch(GROUPS_SQL.format(since_filter=""))
    print(f"{len(groups)} groups share court + process number + date", flush=True)

    updates: list[tuple[Optional[str], str]] = []  # (duplicate_of, doc_id)
    marked = kept_distinct = 0
    for group in groups:
        rows = await conn.fetch(
            "SELECT doc_id, duplicate_of, length(full_text) AS text_len, "
            "       left(full_text, $2) AS head, created_at, "
            "       metadata->>'extraction_confidence' AS confidence "
            "FROM documents WHERE doc_id = ANY($1::text[])",
            group["doc_ids"], COMPARE_CHARS * 2,
        )
        rows = sorted(rows, key=lambda r: (
            -(r["text_len"] or 0),
            CONFIDENCE_ORDER.get(r["confidence"], 3),
            r["created_at"],
            r["doc_id"],
        ))
        canonical = rows[0]
        canonical_head = _normalized(canonical["head"])
        if canonical["duplicate_of"] is not None:
            updates.append((None, canonical["doc_id"]))
        for row in rows[1:]:
            target = canonical["doc_id"] if _similar(canonical_head, _normalized(row["head"])) else None
            if target:
                marked += 1
            else:
                kept_distinct += 1
            if row["duplicate_of"] != target:
                updates.append((target, row["doc_id"]))

    print(f"duplicates: {marked} | same key but different text (kept): {kept_distinct} | "
          f"rows to update: {len(updates)}", flush=True)
    if updates and not dry_run:
        await conn.executemany(
            "UPDATE documents SET duplicate_of = $1 WHERE doc_id = $2", updates
        )
        print("Updated.")
    if not dry_run:
        # Stale statistics make the planner skip the HNSW index on filtered
        # vector searches (see the 20261010 migration).
        await conn.execute("ANALYZE documents (duplicate_of)")
    total = await conn.fetchval("SELECT count(*) FROM documents WHERE duplicate_of IS NOT NULL")
    print(f"Documents marked as duplicates in total: {total}")
    await conn.close()


@click.command()
@click.option("--updated-since", type=click.DateTime(), default=None,
              help="Only re-check groups containing documents written since this date")
@click.option("--dry-run", is_flag=True)
def main(updated_since: Optional[datetime], dry_run: bool) -> None:
    asyncio.run(run(updated_since, dry_run))


if __name__ == "__main__":
    main()
