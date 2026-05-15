"""Backfill doc_count in legislation_embeddings.

Strategy: process one law at a time using the GIN index on metadata to
quickly find documents that cite each law, avoiding a full-table scan.
This keeps each query fast and avoids statement timeouts.
"""
import asyncio
import os
import time

import asyncpg
from dotenv import load_dotenv

load_dotenv('.env.local')


async def main() -> None:
    conn = await asyncpg.connect(
        host=os.getenv('SUPABASE_DB_HOST'),
        port=int(os.getenv('SUPABASE_DB_PORT', '5432')),
        user=os.getenv('SUPABASE_DB_USER'),
        password=os.getenv('SUPABASE_DB_PASSWORD'),
        database=os.getenv('SUPABASE_DB_NAME', 'postgres'),
        command_timeout=300,  # 5 min per query
    )
    await conn.execute("SET statement_timeout = '300000'")  # 5 min

    # 1. Get all distinct laws in legislation_embeddings
    laws = [r['law'] for r in await conn.fetch(
        "SELECT DISTINCT law FROM legislation_embeddings ORDER BY law"
    )]
    print(f"Found {len(laws)} distinct laws to process")

    total_updated = 0
    t0 = time.time()

    for i, law in enumerate(laws, 1):
        # Use GIN index to get documents citing this law, then count per article
        counts = await conn.fetch(
            """
            SELECT l->>'article' AS article, COUNT(*) AS cnt
            FROM documents,
                 jsonb_array_elements(metadata->'legislation_cited') AS l
            WHERE metadata @> jsonb_build_object(
                    'legislation_cited', jsonb_build_array(jsonb_build_object('law', $1::text))
                  )
              AND l->>'law' = $1
            GROUP BY l->>'article'
            """,
            law,
        )

        if not counts:
            continue

        # Bulk-update all articles for this law
        for row in counts:
            article = row['article'] if row['article'] and row['article'].strip() else None
            await conn.execute(
                """
                UPDATE legislation_embeddings
                SET doc_count = $1
                WHERE law = $2 AND article IS NOT DISTINCT FROM $3
                """,
                row['cnt'], law, article,
            )
            total_updated += 1

        elapsed = time.time() - t0
        print(f"  [{i}/{len(laws)}] {law!r}: {len(counts)} articles updated — "
              f"{total_updated} total, {elapsed:.0f}s elapsed")

    print(f"\nDone: {total_updated} rows updated in {time.time() - t0:.0f}s")

    # Show top 10 most-cited after update
    rows = await conn.fetch(
        "SELECT law, article, doc_count FROM legislation_embeddings "
        "ORDER BY doc_count DESC NULLS LAST LIMIT 10"
    )
    print("\nTop 10 most-cited legislation after backfill:")
    for r in rows:
        print(f"  {r['law']} | art. {r['article']} | {r['doc_count']} docs")

    await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
