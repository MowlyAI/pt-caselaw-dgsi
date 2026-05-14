import asyncio, asyncpg, os
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
    # Enable pgvector extension if not already enabled
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS legislation_embeddings (
            id SERIAL PRIMARY KEY,
            law TEXT NOT NULL,
            article TEXT,
            citation_text TEXT NOT NULL,
            embedding halfvec(1024),
            doc_count INT DEFAULT 0,
            UNIQUE(law, article)
        )
    """)

    # HNSW index for fast cosine similarity search
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_leg_emb_hnsw
        ON legislation_embeddings
        USING hnsw (embedding halfvec_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # GIN index for law-only filtering
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_leg_emb_law
        ON legislation_embeddings (law)
    """)

    print("Table legislation_embeddings created.")
    await conn.close()

if __name__ == '__main__':
    asyncio.run(main())
