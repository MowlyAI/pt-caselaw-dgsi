"""Create and backfill document_legislation_citations in date batches.

Run from the repo root after configuring database env vars:
    python scripts/create_legislation_citations_table.py --start-year 1932 --end-year 2026
"""
from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Iterable

import asyncpg
from dotenv import load_dotenv


TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.document_legislation_citations (
    doc_id text NOT NULL REFERENCES public.documents(doc_id) ON DELETE CASCADE,
    law text NOT NULL,
    article text NOT NULL DEFAULT '',
    decision_date date,
    court_short text,
    is_auj boolean,
    legal_domain text,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, law, article)
)
"""

BACKFILL_SQL = """
INSERT INTO public.document_legislation_citations (
    doc_id, law, article, decision_date, court_short, is_auj, legal_domain
)
SELECT DISTINCT
    d.doc_id,
    btrim(el->>'law') AS law,
    COALESCE(NULLIF(btrim(el->>'article'), ''), '') AS article,
    d.decision_date,
    d.court_short,
    d.is_auj,
    d.legal_domain
FROM public.documents d
CROSS JOIN LATERAL jsonb_array_elements(
    COALESCE(d.metadata->'legislation_cited', '[]'::jsonb)
) AS el
WHERE d.decision_date >= $1::date
  AND d.decision_date < $2::date
  AND NULLIF(btrim(el->>'law'), '') IS NOT NULL
ON CONFLICT (doc_id, law, article) DO UPDATE SET
    decision_date = EXCLUDED.decision_date,
    court_short = EXCLUDED.court_short,
    is_auj = EXCLUDED.is_auj,
    legal_domain = EXCLUDED.legal_domain
"""

BACKFILL_NULL_DATES_SQL = """
INSERT INTO public.document_legislation_citations (
    doc_id, law, article, decision_date, court_short, is_auj, legal_domain
)
SELECT DISTINCT
    d.doc_id,
    btrim(el->>'law') AS law,
    COALESCE(NULLIF(btrim(el->>'article'), ''), '') AS article,
    d.decision_date,
    d.court_short,
    d.is_auj,
    d.legal_domain
FROM public.documents d
CROSS JOIN LATERAL jsonb_array_elements(
    COALESCE(d.metadata->'legislation_cited', '[]'::jsonb)
) AS el
WHERE d.decision_date IS NULL
  AND NULLIF(btrim(el->>'law'), '') IS NOT NULL
ON CONFLICT (doc_id, law, article) DO UPDATE SET
    decision_date = EXCLUDED.decision_date,
    court_short = EXCLUDED.court_short,
    is_auj = EXCLUDED.is_auj,
    legal_domain = EXCLUDED.legal_domain
"""

INDEX_SQL = [
    """
    CREATE INDEX IF NOT EXISTS idx_doc_leg_citations_law_date
    ON public.document_legislation_citations (law, decision_date DESC, doc_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_doc_leg_citations_law_article_prefix
    ON public.document_legislation_citations (
        law, article text_pattern_ops, decision_date DESC, doc_id
    )
    WHERE article <> ''
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_doc_leg_citations_doc_id
    ON public.document_legislation_citations (doc_id)
    """,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1932)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--skip-null-dates", action="store_true")
    return parser.parse_args()


async def connect() -> asyncpg.Connection:
    load_dotenv(".env.local")
    return await asyncpg.connect(
        host=os.getenv("SUPABASE_DB_HOST"),
        port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
        user=os.getenv("SUPABASE_DB_USER"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        database=os.getenv("SUPABASE_DB_NAME", "postgres"),
        statement_cache_size=0,
        command_timeout=300,
    )


def year_ranges(start_year: int, end_year: int) -> Iterable[tuple[str, str]]:
    for year in range(start_year, end_year + 1):
        yield f"{year}-01-01", f"{year + 1}-01-01"


async def backfill(conn: asyncpg.Connection, args: argparse.Namespace) -> None:
    await conn.execute(TABLE_SQL)
    if not args.skip_null_dates:
        status = await conn.execute(BACKFILL_NULL_DATES_SQL)
        print(f"Backfilled documents with null decision_date: {status}")
    for start_date, end_date in year_ranges(args.start_year, args.end_year):
        status = await conn.execute(BACKFILL_SQL, start_date, end_date)
        print(f"Backfilled documents [{start_date}, {end_date}): {status}")


async def create_indexes(conn: asyncpg.Connection) -> None:
    for sql in INDEX_SQL:
        await conn.execute(sql)
    await conn.execute("ANALYZE public.document_legislation_citations")


async def main() -> None:
    args = parse_args()
    conn = await connect()
    try:
        await backfill(conn, args)
        await create_indexes(conn)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())