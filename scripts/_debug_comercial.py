import asyncio, asyncpg, os, json
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

    # What articles does Código Comercial have in embeddings near 20?
    rows2 = await conn.fetch("""
        SELECT law, article, doc_count FROM legislation_embeddings
        WHERE law = 'Código Comercial' AND article IS NOT NULL
        ORDER BY doc_count DESC NULLS LAST LIMIT 15
    """)
    print('Top Código Comercial articles in embeddings:')
    for r in rows2:
        print(f"  art. {r['article']} | {r['doc_count']} docs")

    # What exact article values exist for '20' in Código Comercial?
    rows3 = await conn.fetch("""
        SELECT law, article, doc_count FROM legislation_embeddings
        WHERE law = 'Código Comercial' AND article LIKE '%20%'
        ORDER BY doc_count DESC NULLS LAST LIMIT 10
    """)
    print('\nCódigo Comercial articles containing "20":')
    for r in rows3:
        print(f"  art. {r['article']} | {r['doc_count']} docs")

    # Check Código do Notariado
    rows4 = await conn.fetch("""
        SELECT law, article, doc_count FROM legislation_embeddings
        WHERE law ILIKE '%Notariado%'
        ORDER BY doc_count DESC NULLS LAST LIMIT 5
    """)
    print('\nCódigo do Notariado articles:')
    for r in rows4:
        print(f"  {r['law']} | art. {r['article']} | {r['doc_count']} docs")

    # Count how many docs cite Código do Notariado (law-only)
    import json as _json
    import time
    t0 = time.time()
    count = await conn.fetchval("""
        SELECT COUNT(*) FROM documents
        WHERE metadata @> '{"legislation_cited": [{"law": "Código do Notariado"}]}'::jsonb
    """)
    print(f'\nCódigo do Notariado law-only: {count} docs in {time.time()-t0:.2f}s')

    await conn.close()

asyncio.run(main())
