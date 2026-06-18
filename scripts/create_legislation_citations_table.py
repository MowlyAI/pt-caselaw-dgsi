"""Create and backfill document_legislation_citations in batches.

Run from the repo root after configuring database env vars:
    python scripts/create_legislation_citations_table.py --batch-size 5000
"""
from __future__ import annotations

import argparse
import asyncio
import os

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
WHERE d.id > $1
  AND d.id <= $2
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
    parser.add_argument("--batch-size", type=int, default=5000)
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


async def backfill(conn: asyncpg.Connection, batch_size: int) -> None:
    max_id = await conn.fetchval("SELECT COALESCE(max(id), 0) FROM public.documents")
    await conn.execute(TABLE_SQL)
    for start in range(0, max_id, batch_size):
        end = start + batch_size
        status = await conn.execute(BACKFILL_SQL, start, end)
        print(f"Backfilled documents id ({start}, {end}]: {status}")


async def create_indexes(conn: asyncpg.Connection) -> None:
    for sql in INDEX_SQL:
        await conn.execute(sql)
    await conn.execute("ANALYZE public.document_legislation_citations")


async def main() -> None:
    args = parse_args()
    conn = await connect()
    try:
        await backfill(conn, args.batch_size)
        await create_indexes(conn)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())