"""Fix malformed article values in legislation_embeddings.

Patterns fixed:
  art342.º  → 342.º
  art.º 334.º → 334.º
  art483.º  → 483.º
  art69     → 69

Deduplication: if canonical already exists for (law, canonical_article),
merge doc_counts and delete the malformed row; otherwise update in-place.
"""
import asyncio
import os
import re

import asyncpg
from dotenv import load_dotenv

load_dotenv('.env.local')


def _is_malformed(article: str | None) -> bool:
    """Return True if article starts with a stray 'art' prefix."""
    if not article:
        return False
    return bool(re.match(r'^art[0-9\.]', article, re.IGNORECASE))


def _canonical(article: str) -> str:
    """Strip the stray 'art' / 'art.' / 'art.º ' prefix."""
    m = re.match(r'^art\.?(?:º\s*)?(.+)$', article, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return article


def _citation_text(law: str, article: str | None) -> str:
    if article and article.strip():
        return f"{law} art. {article.strip()}"
    return law.strip()


async def main() -> None:
    conn = await asyncpg.connect(
        host=os.getenv('SUPABASE_DB_HOST'),
        port=int(os.getenv('SUPABASE_DB_PORT', '5432')),
        user=os.getenv('SUPABASE_DB_USER'),
        password=os.getenv('SUPABASE_DB_PASSWORD'),
        database=os.getenv('SUPABASE_DB_NAME', 'postgres'),
    )
    await conn.execute("SET statement_timeout = '0'")

    # --- 1. Fetch all rows with malformed articles ---
    rows = await conn.fetch(
        "SELECT id, law, article, doc_count FROM legislation_embeddings "
        "WHERE article ~ '^art[0-9\\.]'"
    )
    print(f"Found {len(rows)} malformed article rows")
    if not rows:
        print("Nothing to fix.")
        await conn.close()
        return

    deleted = 0
    updated = 0
    skipped = 0

    for row in rows:
        rid = row['id']
        law = row['law']
        article = row['article']
        doc_count = row['doc_count'] or 0
        canon = _canonical(article)

        if canon == article:
            skipped += 1
            continue  # nothing changed

        # Check if canonical already exists
        existing = await conn.fetchrow(
            "SELECT id, doc_count FROM legislation_embeddings WHERE law = $1 AND article IS NOT DISTINCT FROM $2",
            law, canon,
        )

        if existing:
            # Merge: keep canonical row, delete malformed, sum doc_counts
            merged_count = (existing['doc_count'] or 0) + doc_count
            await conn.execute(
                "UPDATE legislation_embeddings SET doc_count = $1 WHERE id = $2",
                merged_count, existing['id'],
            )
            await conn.execute("DELETE FROM legislation_embeddings WHERE id = $1", rid)
            deleted += 1
        else:
            # Update in-place
            new_ct = _citation_text(law, canon)
            await conn.execute(
                "UPDATE legislation_embeddings SET article = $1, citation_text = $2 WHERE id = $3",
                canon, new_ct, rid,
            )
            updated += 1

        if (deleted + updated) % 500 == 0:
            print(f"  Progress: {deleted} deleted, {updated} updated, {skipped} skipped")

    print(f"\nDone: {deleted} deleted (merged), {updated} updated in-place, {skipped} skipped")

    # --- 2. Verify ---
    remaining = await conn.fetchval(
        "SELECT COUNT(*) FROM legislation_embeddings WHERE article ~ '^art[0-9\\.]'"
    )
    print(f"Remaining malformed rows: {remaining}")

    # Show top examples after fix
    sample = await conn.fetch(
        "SELECT law, article, doc_count FROM legislation_embeddings "
        "WHERE law = 'Código Civil' ORDER BY doc_count DESC NULLS LAST LIMIT 5"
    )
    print("\nCódigo Civil top articles after fix:")
    for r in sample:
        print(f"  {r['law']} | art. {r['article']} | cited {r['doc_count']} times")

    await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
