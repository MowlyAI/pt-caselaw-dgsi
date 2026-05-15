"""Debug why 'codigo comercial artigo 20' returns wrong documents."""
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

    # What does the semantic endpoint select for "codigo comercial artigo 20"?
    # The LLM selects "Código Comercial art. 20.º"
    # The exact-match query is: metadata @> '{"legislation_cited": [{"law": "Código Comercial", "article": "20.º"}]}'
    match_obj = json.dumps({"legislation_cited": [{"law": "Código Comercial", "article": "20.º"}]})
    rows = await conn.fetch(
        f"SELECT doc_id, metadata->'legislation_cited' as leg FROM documents WHERE metadata @> $1::jsonb LIMIT 5",
        match_obj,
    )
    print(f"Docs matching Código Comercial art. 20.º: {len(rows)}")
    for r in rows:
        leg = json.loads(r['leg'])
        cc = [l for l in leg if 'Comercial' in l.get('law', '')]
        print(f"  doc {r['doc_id'][:12]}: Comercial cites: {cc}")

    # What if LLM selected differently? Check with just "20"
    match_obj2 = json.dumps({"legislation_cited": [{"law": "Código Comercial", "article": "20"}]})
    rows2 = await conn.fetch(
        f"SELECT doc_id FROM documents WHERE metadata @> $1::jsonb LIMIT 5",
        match_obj2,
    )
    print(f"\nDocs matching Código Comercial art. 20 (no º): {len(rows2)}")

    # Show all article variants for CC art 20 in embeddings
    rows3 = await conn.fetch("""
        SELECT law, article, doc_count FROM legislation_embeddings
        WHERE law = 'Código Comercial' AND (article LIKE '20%' OR article LIKE '%20.%')
        ORDER BY doc_count DESC NULLS LAST
    """)
    print("\nCódigo Comercial '20' article variants in embeddings:")
    for r in rows3:
        print(f"  art. {r['article']!r} | {r['doc_count']} docs")

    await conn.close()

asyncio.run(main())
