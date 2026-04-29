"""FastAPI app for hybrid search over DGSI Portuguese caselaw.

Uses Supabase Postgres directly via asyncpg:
  * HNSW indexes on `embedding` / `embedding_context` / `embedding_ratio`
    (halfvec_cosine_ops) for vector search.
  * GIN index on `fts` (tsvector built with unaccent + portuguese) for FTS.
"""
import asyncio
import json as _json
import os
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, Optional

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
    """Per-connection setup: tune HNSW ef_search and enable iterative scan
    so filtered semantic queries still return enough results."""
    await conn.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}")
    # pgvector >= 0.8: keep walking the HNSW graph when WHERE filters
    # would otherwise leave the LIMIT under-filled.
    await conn.execute("SET hnsw.iterative_scan = strict_order")


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


class Filters(BaseModel):
    """Optional filters applied to every search variant."""
    court: Optional[list[str]] = None
    legal_domain: Optional[str] = None  # ILIKE substring match
    is_auj: Optional[bool] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    decision_type: Optional[list[str]] = None  # metadata->>'decision_type'
    extraction_confidence: Optional[list[str]] = None


class SearchResponse(BaseModel):
    query: str
    query_semantic: Optional[str] = None
    query_keywords: Optional[str] = None
    count: int
    embedding_field: str
    mode: str
    filters: Optional[Filters] = None
    results: list[SearchResult]


def _build_filters(f: Optional[Filters], start_idx: int = 1) -> tuple[str, list[Any]]:
    """Build a SQL WHERE fragment (without 'WHERE'/'AND' prefix) and the
    matching positional params list. The first placeholder will be `$start_idx`.
    Returns ("", []) when no filters are set.
    """
    if f is None:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    idx = start_idx
    if f.court:
        clauses.append(f"court_short = ANY(${idx}::text[])")
        params.append(f.court)
        idx += 1
    if f.legal_domain:
        clauses.append(f"legal_domain ILIKE ${idx}")
        params.append(f"%{f.legal_domain}%")
        idx += 1
    if f.is_auj is not None:
        clauses.append(f"is_auj = ${idx}")
        params.append(f.is_auj)
        idx += 1
    if f.date_from is not None:
        clauses.append(f"decision_date >= ${idx}")
        params.append(f.date_from)
        idx += 1
    if f.date_to is not None:
        clauses.append(f"decision_date <= ${idx}")
        params.append(f.date_to)
        idx += 1
    if f.decision_type:
        clauses.append(f"metadata->>'decision_type' = ANY(${idx}::text[])")
        params.append(f.decision_type)
        idx += 1
    if f.extraction_confidence:
        clauses.append(f"metadata->>'extraction_confidence' = ANY(${idx}::text[])")
        params.append(f.extraction_confidence)
        idx += 1
    return " AND ".join(clauses), params


def _filters_from_query(
    court: Optional[list[str]],
    legal_domain: Optional[str],
    is_auj: Optional[bool],
    date_from: Optional[date],
    date_to: Optional[date],
    decision_type: Optional[list[str]],
    extraction_confidence: Optional[list[str]],
) -> Optional[Filters]:
    """Build a Filters model from raw FastAPI Query params, returning None
    when nothing was set so call sites can short-circuit."""
    if not any([court, legal_domain, is_auj is not None, date_from, date_to,
                decision_type, extraction_confidence]):
        return None
    return Filters(
        court=court or None,
        legal_domain=legal_domain,
        is_auj=is_auj,
        date_from=date_from,
        date_to=date_to,
        decision_type=decision_type or None,
        extraction_confidence=extraction_confidence or None,
    )


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


async def _search_semantic(q: str, field: str, limit: int,
                           filters: Optional[Filters] = None
                           ) -> list[tuple[str, float]]:
    """Vector search using the HNSW index on `field`. Returns [(doc_id, similarity)]."""
    emb = await embed_query(q)
    # Format as a pgvector text literal: '[0.1,0.2,...]'
    emb_lit = "[" + ",".join(f"{x:.7f}" for x in emb) + "]"
    # $1 = embedding, filter params start at $2, limit is the last param.
    filt_sql, filt_params = _build_filters(filters, start_idx=2)
    where = f"{field} IS NOT NULL"
    if filt_sql:
        where += f" AND {filt_sql}"
    limit_idx = 2 + len(filt_params)
    sql = (
        f"SELECT doc_id, (1 - ({field} <=> $1::halfvec))::real AS sim "
        f"FROM documents "
        f"WHERE {where} "
        f"ORDER BY {field} <=> $1::halfvec "
        f"LIMIT ${limit_idx}"
    )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, emb_lit, *filt_params, limit)
    return [(r["doc_id"], float(r["sim"])) for r in rows]


FTS_CANDIDATE_CAP = int(os.getenv("FTS_CANDIDATE_CAP", "1500"))


async def _search_fts(q: str, limit: int,
                      filters: Optional[Filters] = None
                      ) -> list[tuple[str, float]]:
    """Full-text search via the GIN `fts` index.

    Broad queries can match >100k rows; ranking each one would force a heap
    fetch over the full tsvector column (multi-second). We cap the candidate
    set to FTS_CANDIDATE_CAP rows (in index order), then rank within that.
    """
    # $1 = q, filter params start at $2, then $limit_idx, then $cap_idx.
    filt_sql, filt_params = _build_filters(filters, start_idx=2)
    where = "fts @@ websearch_to_tsquery('portuguese', $1)"
    if filt_sql:
        where += f" AND {filt_sql}"
    limit_idx = 2 + len(filt_params)
    cap_idx = limit_idx + 1
    sql = (
        "WITH cands AS ("
        "  SELECT doc_id, fts "
        "    FROM documents "
        f"   WHERE {where} "
        f"   LIMIT ${cap_idx}"
        ") "
        "SELECT doc_id, "
        "       ts_rank_cd(fts, websearch_to_tsquery('portuguese', $1))::real AS rank "
        "  FROM cands "
        " ORDER BY rank DESC "
        f" LIMIT ${limit_idx}"
    )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, q, *filt_params, limit, FTS_CANDIDATE_CAP)
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


# Common filter Query parameter declarations (kept here so /search variants
# share an identical signature).
_Q_COURT = Query(None, description="Filter by court_short (repeat for OR). e.g. STJ, TRP")
_Q_LEGAL_DOMAIN = Query(None, description="Substring match (ILIKE) on legal_domain")
_Q_IS_AUJ = Query(None, description="Filter for AUJ (jurisprudence unification) decisions")
_Q_DATE_FROM = Query(None, description="decision_date >= (YYYY-MM-DD)")
_Q_DATE_TO = Query(None, description="decision_date <= (YYYY-MM-DD)")
_Q_DECISION_TYPE = Query(None, description="Filter by metadata.decision_type (repeat for OR)")
_Q_EXTRACTION_CONFIDENCE = Query(None, description="Filter by extraction_confidence (high/medium/low)")


def _resolve_queries(q: Optional[str], q_semantic: Optional[str],
                     q_keywords: Optional[str], need_sem: bool, need_fts: bool
                     ) -> tuple[Optional[str], Optional[str]]:
    """Pick the effective semantic / keyword strings, falling back to `q`.
    Raises 400 if any required slot ends up empty."""
    sem = q_semantic if q_semantic is not None else q
    kw = q_keywords if q_keywords is not None else q
    if need_sem and not sem:
        raise HTTPException(400, "Provide `q` or `q_semantic` for semantic search")
    if need_fts and not kw:
        raise HTTPException(400, "Provide `q` or `q_keywords` for keyword search")
    return sem, kw


@app.get("/search/semantic", response_model=SearchResponse, tags=["search"])
async def search_semantic(
    q: Optional[str] = Query(None, description="Shared query (used if q_semantic is omitted)"),
    q_semantic: Optional[str] = Query(None, description="Text to embed for vector search"),
    limit: int = Query(20, ge=1, le=100),
    embedding_field: str = Query("embedding"),
    court: Optional[list[str]] = _Q_COURT,
    legal_domain: Optional[str] = _Q_LEGAL_DOMAIN,
    is_auj: Optional[bool] = _Q_IS_AUJ,
    date_from: Optional[date] = _Q_DATE_FROM,
    date_to: Optional[date] = _Q_DATE_TO,
    decision_type: Optional[list[str]] = _Q_DECISION_TYPE,
    extraction_confidence: Optional[list[str]] = _Q_EXTRACTION_CONFIDENCE,
):
    """Pure semantic search using a Postgres HNSW index."""
    if embedding_field not in VALID_FIELDS:
        raise HTTPException(400, f"embedding_field must be one of {sorted(VALID_FIELDS)}")
    sem_q, _ = _resolve_queries(q, q_semantic, None, need_sem=True, need_fts=False)
    filters = _filters_from_query(court, legal_domain, is_auj, date_from, date_to,
                                  decision_type, extraction_confidence)
    hits = await _search_semantic(sem_q, embedding_field, limit, filters)
    docs = await _fetch_docs([doc_id for doc_id, _ in hits])
    results = [
        SearchResult(similarity=round(sim, 4), **docs[doc_id])
        for doc_id, sim in hits if doc_id in docs
    ]
    return SearchResponse(query=sem_q, query_semantic=sem_q, count=len(results),
                          embedding_field=embedding_field, mode="semantic",
                          filters=filters, results=results)


@app.get("/search/fts", response_model=SearchResponse, tags=["search"])
async def search_fts(
    q: Optional[str] = Query(None, description="Shared query (used if q_keywords is omitted)"),
    q_keywords: Optional[str] = Query(None, description="Keyword query for FTS"),
    limit: int = Query(20, ge=1, le=100),
    court: Optional[list[str]] = _Q_COURT,
    legal_domain: Optional[str] = _Q_LEGAL_DOMAIN,
    is_auj: Optional[bool] = _Q_IS_AUJ,
    date_from: Optional[date] = _Q_DATE_FROM,
    date_to: Optional[date] = _Q_DATE_TO,
    decision_type: Optional[list[str]] = _Q_DECISION_TYPE,
    extraction_confidence: Optional[list[str]] = _Q_EXTRACTION_CONFIDENCE,
):
    """Full-text search via the GIN `fts` index (portuguese + unaccent)."""
    _, kw_q = _resolve_queries(q, None, q_keywords, need_sem=False, need_fts=True)
    filters = _filters_from_query(court, legal_domain, is_auj, date_from, date_to,
                                  decision_type, extraction_confidence)
    hits = await _search_fts(kw_q, limit, filters)
    docs = await _fetch_docs([doc_id for doc_id, _ in hits])
    results = [
        SearchResult(fts_rank=round(rank, 4), **docs[doc_id])
        for doc_id, rank in hits if doc_id in docs
    ]
    return SearchResponse(query=kw_q, query_keywords=kw_q, count=len(results),
                          embedding_field="-", mode="fts",
                          filters=filters, results=results)


@app.get("/search", response_model=SearchResponse, tags=["search"])
async def search_hybrid(
    q: Optional[str] = Query(None, description="Shared query, used as fallback for both q_semantic and q_keywords"),
    q_semantic: Optional[str] = Query(None, description="Text to embed for vector search (defaults to q)"),
    q_keywords: Optional[str] = Query(None, description="Keyword query for FTS (defaults to q)"),
    limit: int = Query(20, ge=1, le=100),
    embedding_field: str = Query("embedding"),
    rrf_k: int = Query(50, ge=1, description="Reciprocal Rank Fusion constant"),
    fts_weight: float = Query(1.0, ge=0, le=10),
    vector_weight: float = Query(1.0, ge=0, le=10),
    court: Optional[list[str]] = _Q_COURT,
    legal_domain: Optional[str] = _Q_LEGAL_DOMAIN,
    is_auj: Optional[bool] = _Q_IS_AUJ,
    date_from: Optional[date] = _Q_DATE_FROM,
    date_to: Optional[date] = _Q_DATE_TO,
    decision_type: Optional[list[str]] = _Q_DECISION_TYPE,
    extraction_confidence: Optional[list[str]] = _Q_EXTRACTION_CONFIDENCE,
):
    """Hybrid: HNSW + GIN FTS merged with RRF.

    Pass `q_semantic` and `q_keywords` separately to drive the vector and
    keyword sides independently, or fall back to a single `q` for both.
    """
    if embedding_field not in VALID_FIELDS:
        raise HTTPException(400, f"embedding_field must be one of {sorted(VALID_FIELDS)}")
    sem_q, kw_q = _resolve_queries(q, q_semantic, q_keywords,
                                   need_sem=True, need_fts=True)
    filters = _filters_from_query(court, legal_domain, is_auj, date_from, date_to,
                                  decision_type, extraction_confidence)
    sem, fts = await asyncio.gather(
        _search_semantic(sem_q, embedding_field, limit * 4, filters),
        _search_fts(kw_q, limit * 4, filters),
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
    # `query` echoes the primary string, prioritising semantic for clarity.
    return SearchResponse(query=sem_q or kw_q, query_semantic=sem_q,
                          query_keywords=kw_q, count=len(results),
                          embedding_field=embedding_field, mode="hybrid",
                          filters=filters, results=results)


FILTERS_CACHE_TTL = int(os.getenv("FILTERS_CACHE_TTL", "3600"))  # seconds
_filters_cache: dict[str, tuple[float, dict]] = {}


async def _compute_filters_payload(legal_domain_prefix: Optional[str],
                                   top_legal_domains: int) -> dict:
    """Run the filter-discovery queries and shape the response."""
    async with db_pool.acquire() as conn:
        court_rows, is_auj_rows, date_row, ld_total = await asyncio.gather(
            conn.fetch(
                "SELECT court_short AS value, count(*) AS count FROM documents "
                " WHERE court_short IS NOT NULL "
                " GROUP BY court_short ORDER BY count DESC"
            ),
            conn.fetch(
                "SELECT is_auj AS value, count(*) AS count FROM documents "
                " WHERE is_auj IS NOT NULL "
                " GROUP BY is_auj ORDER BY value DESC"
            ),
            conn.fetchrow(
                "SELECT min(decision_date) AS min, max(decision_date) AS max, "
                "       count(decision_date) AS count "
                "  FROM documents"
            ),
            conn.fetchval(
                "SELECT count(distinct legal_domain) FROM documents "
                " WHERE legal_domain IS NOT NULL"
            ),
        )
        if legal_domain_prefix:
            ld_rows = await conn.fetch(
                "SELECT legal_domain AS value, count(*) AS count FROM documents "
                " WHERE legal_domain ILIKE $1 "
                " GROUP BY legal_domain ORDER BY count DESC LIMIT $2",
                f"%{legal_domain_prefix}%", top_legal_domains,
            )
        else:
            ld_rows = await conn.fetch(
                "SELECT legal_domain AS value, count(*) AS count FROM documents "
                " WHERE legal_domain IS NOT NULL "
                " GROUP BY legal_domain ORDER BY count DESC LIMIT $1",
                top_legal_domains,
            )
        # Metadata-derived enums. The jsonb GIN index helps with `?` lookups.
        dt_rows, conf_rows = await asyncio.gather(
            conn.fetch(
                "SELECT metadata->>'decision_type' AS value, count(*) AS count "
                "  FROM documents WHERE metadata ? 'decision_type' "
                " GROUP BY metadata->>'decision_type' ORDER BY count DESC"
            ),
            conn.fetch(
                "SELECT metadata->>'extraction_confidence' AS value, count(*) AS count "
                "  FROM documents WHERE metadata ? 'extraction_confidence' "
                " GROUP BY metadata->>'extraction_confidence' ORDER BY count DESC"
            ),
        )
    return {
        "court": [{"value": r["value"], "count": r["count"]} for r in court_rows],
        "is_auj": [{"value": r["value"], "count": r["count"]} for r in is_auj_rows],
        "decision_date": {
            "min": date_row["min"].isoformat() if date_row["min"] else None,
            "max": date_row["max"].isoformat() if date_row["max"] else None,
            "count": date_row["count"],
        },
        "legal_domain": {
            "distinct_count": ld_total,
            "top": [{"value": r["value"], "count": r["count"]} for r in ld_rows],
            "note": "High-cardinality field; use `legal_domain_prefix` to autocomplete.",
        },
        "decision_type": [{"value": r["value"], "count": r["count"]} for r in dt_rows],
        "extraction_confidence": [
            {"value": r["value"], "count": r["count"]} for r in conf_rows
        ],
    }


@app.get("/filters", tags=["info"])
async def get_filters(
    legal_domain_prefix: Optional[str] = Query(
        None, description="Substring filter for the legal_domain enumeration"),
    top_legal_domains: int = Query(
        50, ge=1, le=500,
        description="How many legal_domain values to return (most frequent first)"),
    refresh: bool = Query(False, description="Bypass the in-memory cache"),
):
    """Return the filter fields supported by /search and the unique values
    available for each. Heavy aggregations are cached in-memory for
    `FILTERS_CACHE_TTL` seconds (default 1h)."""
    cache_key = f"{legal_domain_prefix or ''}|{top_legal_domains}"
    now = time.time()
    if not refresh:
        cached = _filters_cache.get(cache_key)
        if cached and (now - cached[0]) < FILTERS_CACHE_TTL:
            return {**cached[1], "cached": True,
                    "cache_age_seconds": int(now - cached[0])}
    payload = await _compute_filters_payload(legal_domain_prefix, top_legal_domains)
    _filters_cache[cache_key] = (now, payload)
    return {**payload, "cached": False, "cache_age_seconds": 0}


@app.get("/document/{doc_id}", tags=["documents"])
async def get_document(doc_id: str):
    docs = await _fetch_docs([doc_id])
    if doc_id not in docs:
        raise HTTPException(404, "Document not found")
    return docs[doc_id]
