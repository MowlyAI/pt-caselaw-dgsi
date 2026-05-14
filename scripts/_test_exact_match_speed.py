import asyncio
import json
import os
import time

import asyncpg
from dotenv import load_dotenv

load_dotenv('.env.local')


async def main():
    conn = await asyncpg.connect(
        host=os.getenv('SUPABASE_DB_HOST'),
        port=int(os.getenv('SUPABASE_DB_PORT', '5432')),
        user=os.getenv('SUPABASE_DB_USER'),
        password=os.getenv('SUPABASE_DB_PASSWORD'),
        database=os.getenv('SUPABASE_DB_NAME', 'postgres'),
    )

    tests = [
        ('CE art. 2', 'Código da Estrada', '2.º'),
        ('CC art. 342', 'Código Civil', '342.º'),
        ('CC law-only', 'Código Civil', None),
        ('CT art. 394', 'Código do Trabalho', '394.º'),
    ]

    for label, law, article in tests:
        t0 = time.time()
        if article:
            match = {"legislation_cited": [{"law": law, "article": article}]}
        else:
            match = {"legislation_cited": [{"law": law}]}
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE metadata @> $1::jsonb",
            json.dumps(match),
        )
        t1 = time.time()
        print(f'{label}: {count} docs in {t1-t0:.2f}s')

    # Also test old LIKE approach for comparison
    t0 = time.time()
    count = await conn.fetchval(
        """
        SELECT COUNT(*) FROM documents
        WHERE EXISTS (
            SELECT 1 FROM jsonb_array_elements(metadata->'legislation_cited') AS lc
            WHERE lc->>'law' = 'Código Civil' AND lc->>'article' ILIKE '342.º%'
        )
        """
    )
    t1 = time.time()
    print(f'CC art. 342 (old LIKE): {count} docs in {t1-t0:.2f}s')

    await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
