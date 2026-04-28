"""FastAPI app for hybrid search over DGSI Portuguese caselaw.

Uses Supabase Postgres directly via asyncpg:
  * HNSW indexes on `embedding` / `embedding_context` / `embedding_ratio`
    (halfvec_cosine_ops) for vector search.
  * GIN index on `fts` (tsvector built with unaccent + portuguese) for FTS.
"""
import asyncio
import json as _json
import os
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

import asyncpg
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

load_dotenv(".env.local")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

DB_HOST = os.getenv("SUPABASE_DB_HOST", "")
DB_PORT = int(os.getenv("SUPABASE_DB_PORT", "5432"))
DB_USER = os.getenv("SUPABASE_DB_USER", "")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")

# HNSW search-time recall/speed tradeoff. 40 is a good default; raise for recall.
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "40"))

VALID_FIELDS = {"embedding", "embedding_context", "embedding_ratio"}

DOC_COLUMNS = (
    "doc_id, url, court_short, process_number, decision_date, "
    "legal_domain, is_auj, summary, metadata"
)

# Initialised at startup
db_pool: Optional[asyncpg.Pool] = None
http_client: Optional[httpx.AsyncClient] = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Per-connection setup: tune HNSW ef_search."""
    await conn.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, db_pool
    http_client = httpx.AsyncClient(timeout=30)
    db_pool = await asyncpg.create_pool(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASSWORD, database=DB_NAME,
        min_size=1, max_size=10,
        statement_cache_size=0,  # required for the Supabase pgbouncer pooler
        command_timeout=30,
        init=_init_connection,
    )
    # Probe the connection so we fail fast on bad credentials.
    async with db_pool.acquire() as conn:
        n = await conn.fetchval("SELECT count(*) FROM documents WHERE embedding IS NOT NULL")
        print(f"Connected to Postgres. {n} documents with embeddings.")
    yield
    await db_pool.close()
    await http_client.aclose()


app = FastAPI(
    title="PT Caselaw DGSI Search API",
    description="Hybrid search over Portuguese court decisions (STJ, STA, TR, TCA, ...).",
    version="3.0.0",
    lifespan=lifespan,
)


class SearchResult(BaseModel):
    doc_id: str
    url: str
    court_short: str
    process_number: Optional[str] = None
    decision_date: Optional[date] = None
    legal_domain: Optional[str] = None
    is_auj: Optional[bool] = None
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    similarity: Optional[float] = None
    fts_rank: Optional[float] = None
    hybrid_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    count: int
    embedding_field: str
    mode: str
    results: list[SearchResult]


async def embed_query(text: str) -> list[float]:
    payload = {"model": EMBEDDING_MODEL, "input": text, "dimensions": EMBEDDING_DIM}
    resp = await http_client.post(
        f"{OPENROUTER_BASE}/embeddings",
        json=payload,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
    )
    if resp.status_code != 200:
        raise HTTPException(502, f"Embedding API error: {resp.status_code} {resp.text[:200]}")
    return resp.json()["data"][0]["embedding"]


def _row_to_dict(row: asyncpg.Record) -> dict:
    """Convert an asyncpg Record into a JSON-serialisable dict for SearchResult."""
    d = dict(row)
    md = d.get("metadata")
    if isinstance(md, str):
        # asyncpg returns jsonb as str when no codec is registered.
        try:
            d["metadata"] = _json.loads(md)
        except Exception:
            d["metadata"] = None
    return d


async def _fetch_docs(doc_ids: list[str]) -> dict[str, dict]:
    """Fetch metadata for the given doc_ids. Returns {doc_id: row}."""
    if not doc_ids:
        return {}
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT {DOC_COLUMNS} FROM documents WHERE doc_id = ANY($1::text[])",
            doc_ids,
        )
    return {r["doc_id"]: _row_to_dict(r) for r in rows}


async def _search_semantic(q: str, field: str, limit: int) -> list[tuple[str, float]]:
    """Vector search using the HNSW index on `field`. Returns [(doc_id, similarity)]."""
    emb = await embed_query(q)
    # Format as a pgvector text literal: '[0.1,0.2,...]'
    emb_lit = "[" + ",".join(f"{x:.7f}" for x in emb) + "]"
    sql = (
        f"SELECT doc_id, (1 - ({field} <=> $1::halfvec))::real AS sim "
        f"FROM documents "
        f"WHERE {field} IS NOT NULL "
        f"ORDER BY {field} <=> $1::halfvec "
        f"LIMIT $2"
    )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, emb_lit, limit)
    return [(r["doc_id"], float(r["sim"])) for r in rows]


FTS_CANDIDATE_CAP = int(os.getenv("FTS_CANDIDATE_CAP", "1500"))


async def _search_fts(q: str, limit: int) -> list[tuple[str, float]]:
    """Full-text search via the GIN `fts` index.

    Broad queries can match >100k rows; ranking each one would force a heap
    fetch over the full tsvector column (multi-second). We cap the candidate
    set to FTS_CANDIDATE_CAP rows (in index order), then rank within that.
    """
    sql = (
        "WITH cands AS ("
        "  SELECT doc_id, fts "
        "    FROM documents "
        "   WHERE fts @@ websearch_to_tsquery('portuguese', $1) "
        "   LIMIT $3"
        ") "
        "SELECT doc_id, "
        "       ts_rank_cd(fts, websearch_to_tsquery('portuguese', $1))::real AS rank "
        "  FROM cands "
        " ORDER BY rank DESC "
        " LIMIT $2"
    )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, q, limit, FTS_CANDIDATE_CAP)
    return [(r["doc_id"], float(r["rank"])) for r in rows]


def _rrf_merge(semantic: list[tuple[str, float]], fts: list[tuple[str, float]],
               k: int = 50, w_sem: float = 1.0, w_fts: float = 1.0) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(semantic, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + w_sem / (k + rank)
    for rank, (doc_id, _) in enumerate(fts, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + w_fts / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


@app.get("/", tags=["health"])
async def root():
    return {"name": "PT Caselaw DGSI Search API", "status": "ok", "version": "3.0.0"}


@app.get("/health", tags=["health"])
async def health():
    ok = db_pool is not None and http_client is not None
    return {"status": "healthy" if ok else "degraded"}


@app.get("/stats", tags=["info"])
async def stats():
    async with db_pool.acquire() as conn:
        counts = await conn.fetchrow(
            "SELECT count(*) AS total, "
            "       count(*) FILTER (WHERE embedding IS NOT NULL) AS embedding, "
            "       count(*) FILTER (WHERE embedding_context IS NOT NULL) AS embedding_context, "
            "       count(*) FILTER (WHERE embedding_ratio IS NOT NULL) AS embedding_ratio "
            "  FROM documents"
        )
    return {
        "total_documents": counts["total"],
        "embeddings": {
            "embedding": counts["embedding"],
            "embedding_context": counts["embedding_context"],
            "embedding_ratio": counts["embedding_ratio"],
        },
        "embedding_model": EMBEDDING_MODEL,
        "hnsw_ef_search": HNSW_EF_SEARCH,
    }


@app.get("/search/semantic", response_model=SearchResponse, tags=["search"])
async def search_semantic(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    embedding_field: str = Query("embedding"),
):
    """Pure semantic search using a Postgres HNSW index."""
    if embedding_field not in VALID_FIELDS:
        raise HTTPException(400, f"embedding_field must be one of {sorted(VALID_FIELDS)}")
    hits = await _search_semantic(q, embedding_field, limit)
    docs = await _fetch_docs([doc_id for doc_id, _ in hits])
    results = [
        SearchResult(similarity=round(sim, 4), **docs[doc_id])
        for doc_id, sim in hits if doc_id in docs
    ]
    return SearchResponse(query=q, count=len(results), embedding_field=embedding_field,
                          mode="semantic", results=results)


@app.get("/search/fts", response_model=SearchResponse, tags=["search"])
async def search_fts(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
):
    """Full-text search via the GIN `fts` index (portuguese + unaccent)."""
    hits = await _search_fts(q, limit)
    docs = await _fetch_docs([doc_id for doc_id, _ in hits])
    results = [
        SearchResult(fts_rank=round(rank, 4), **docs[doc_id])
        for doc_id, rank in hits if doc_id in docs
    ]
    return SearchResponse(query=q, count=len(results), embedding_field="-",
                          mode="fts", results=results)


@app.get("/search", response_model=SearchResponse, tags=["search"])
async def search_hybrid(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    embedding_field: str = Query("embedding"),
    rrf_k: int = Query(50, ge=1, description="Reciprocal Rank Fusion constant"),
    fts_weight: float = Query(1.0, ge=0, le=10),
    vector_weight: float = Query(1.0, ge=0, le=10),
):
    """Hybrid: HNSW + GIN FTS merged with RRF."""
    if embedding_field not in VALID_FIELDS:
        raise HTTPException(400, f"embedding_field must be one of {sorted(VALID_FIELDS)}")
    sem, fts = await asyncio.gather(
        _search_semantic(q, embedding_field, limit * 4),
        _search_fts(q, limit * 4),
    )
    merged = _rrf_merge(sem, fts, k=rrf_k, w_sem=vector_weight, w_fts=fts_weight)
    docs = await _fetch_docs([doc_id for doc_id, _ in merged[:limit]])
    sem_map = dict(sem)
    fts_map = dict(fts)
    results = [
        SearchResult(
            similarity=round(sem_map.get(doc_id, 0), 4),
            fts_rank=round(fts_map.get(doc_id, 0), 4),
            hybrid_score=round(score, 6),
            **docs[doc_id],
        )
        for doc_id, score in merged[:limit] if doc_id in docs
    ]
    return SearchResponse(query=q, count=len(results), embedding_field=embedding_field,
                          mode="hybrid", results=results)


@app.get("/document/{doc_id}", tags=["documents"])
async def get_document(doc_id: str):
    docs = await _fetch_docs([doc_id])
    if doc_id not in docs:
        raise HTTPException(404, "Document not found")
    return docs[doc_id]
