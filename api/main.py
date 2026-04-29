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
from fastapi import Body, FastAPI, HTTPException, Path as FastAPIPath, Query
from pydantic import BaseModel, Field

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

EMBEDDING_FIELDS: tuple[str, ...] = ("embedding", "embedding_context", "embedding_ratio")
ALL_SOURCES: tuple[str, ...] = (*EMBEDDING_FIELDS, "fts")

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


API_DESCRIPTION = """
Hybrid search over Portuguese court decisions (DGSI: STJ, STA, STJ, TR*, TCA*, TC, …).

## How to use this API (for LLM agents)

The intended workflow is:

1. **Discover what you can filter on** with `GET /filters`. This returns the
   list of supported filter fields and the available values (courts, decision
   types, top legal domains, date range, …). Use this to ground any filter you
   later send.
2. **Search** with `POST /search` (recommended), `POST /search/semantic` or
   `POST /search/fts`. Always prefer `POST /search` unless you have a strong
   reason — it fuses 3 vector representations and full-text search into one
   ranking.
3. **Hydrate** a specific decision with `GET /document/{doc_id}` to retrieve
   the full extracted metadata (parties, citations, ratio decidendi, …).

## Search model

Every document in the index has three independent embeddings of its content:

| Column              | Embedded text                                       | Best for                                       |
|---------------------|-----------------------------------------------------|------------------------------------------------|
| `embedding`         | Concise summary (≤ 300 words)                       | Topical / conceptual queries                   |
| `embedding_context` | Full context (parties, facts, decision, …)         | Fact-pattern or "find similar case" queries    |
| `embedding_ratio`   | Legal reasoning (ratio decidendi, legal question)   | "What rule was applied?" doctrinal queries     |

Plus a Postgres full-text index (`fts`, Portuguese + unaccent) for exact
keyword matching of legal terms, names, statute numbers, etc.

`POST /search` queries **all four sources in parallel** (the query is
embedded once and reused), then fuses the four ranked lists with
**weighted Reciprocal Rank Fusion** (RRF). Each result reports its
per-source similarity scores, per-source rank and the fused `hybrid_score`,
so you can audit *why* a doc was returned.

## Choosing semantic vs keyword text

`SearchRequest` accepts two strings:

* `q_semantic` — natural-language description of what you're looking for
  (e.g. *"despedimento sem justa causa por uso indevido de email corporativo"*).
  This is what gets embedded.
* `q_keywords` — terse keyword query for FTS (e.g. *"despedimento email
  corporativo"*). Supports `websearch_to_tsquery` syntax (quoted phrases,
  `-exclusion`, `OR`).

If you only have one string, send it as `q` and both sides will use it.

## Filters

All search endpoints accept the same `filters` object. It is composable
(AND across fields). See the `Filters` schema for every supported field;
`GET /filters` returns the actual values present in the corpus.

## Tuning

* `weights` — set any source's weight to 0 to disable it (e.g.
  `{"weights": {"fts": 0}}` for vectors-only). Boost a column above 1.0
  to bias the ranking toward that signal.
* `overfetch` — per-source candidates fetched = `limit * overfetch`
  (default 4). Increase for more recall on heavily filtered queries.
* `rrf_k` — Reciprocal Rank Fusion constant. Lower = sharper ranking
  (top-1 dominates), higher = smoother fusion. Default 50 is usually fine.

## Notable identifiers

* `doc_id` — opaque deterministic ID; stable across re-imports.
* `court_short` — e.g. `STJ`, `STA`, `TRP`, `TCAS`. Use as the `court` filter.
* `is_auj` — true ⇔ the decision is itself an *Acórdão de Uniformização /
  Fixação de Jurisprudência* (binding precedent), not just one citing an AUJ.
"""

OPENAPI_TAGS = [
    {"name": "search", "description":
        "Hybrid / semantic / keyword search over the corpus. All endpoints "
        "are POST and accept the same `SearchRequest` body."},
    {"name": "info", "description":
        "Discovery endpoints: list available filters, corpus stats."},
    {"name": "documents", "description":
        "Hydrate a single document by `doc_id`."},
    {"name": "health", "description":
        "Liveness / readiness probes."},
]

app = FastAPI(
    title="PT Caselaw DGSI Search API",
    description=API_DESCRIPTION,
    version="3.0.0",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
    contact={"name": "pt-caselaw-dgsi"},
)


class Filters(BaseModel):
    """Composable filters applied to every search endpoint (combined with AND).

    Send only the fields you want to constrain — every field is optional.
    Use `GET /filters` to discover the valid values present in the corpus.
    """
    court: Optional[list[str]] = Field(
        None,
        description=(
            "Restrict to one or more courts using `court_short` codes "
            "(exact match, ANY-of). Examples: `STJ`, `STA`, `TC`, `TRP`, "
            "`TRL`, `TRC`, `TRG`, `TCAN`, `TCAS`."
        ),
        examples=[["STJ", "TRP"]],
    )
    legal_domain: Optional[str] = Field(
        None,
        description=(
            "Substring match (case-insensitive, ILIKE `%value%`) against the "
            "`legal_domain` column. The corpus has 6000+ distinct values, so "
            "use a substring rather than an exact value (e.g. `insolvencia`)."
        ),
        examples=["insolvencia"],
    )
    is_auj: Optional[bool] = Field(
        None,
        description=(
            "Filter to (or exclude) Acórdãos de Uniformização / Fixação de "
            "Jurisprudência (binding precedent). `true` keeps only AUJs, "
            "`false` excludes them, `null`/omit returns both."
        ),
        examples=[True],
    )
    date_from: Optional[date] = Field(
        None,
        description="Inclusive lower bound on `decision_date` (ISO `YYYY-MM-DD`).",
        examples=["2020-01-01"],
    )
    date_to: Optional[date] = Field(
        None,
        description="Inclusive upper bound on `decision_date` (ISO `YYYY-MM-DD`).",
        examples=["2024-12-31"],
    )
    decision_type: Optional[list[str]] = Field(
        None,
        description=(
            "Restrict to one or more decision types extracted into "
            "`metadata->>'decision_type'` (ANY-of). Examples: `Acórdão`, "
            "`Sentença`, `Despacho`."
        ),
        examples=[["Acórdão"]],
    )
    extraction_confidence: Optional[list[str]] = Field(
        None,
        description=(
            "Restrict to documents whose LLM extraction confidence is in the "
            "given set. Possible values: `high`, `medium`, `low`."
        ),
        examples=[["high", "medium"]],
    )


class SearchWeights(BaseModel):
    """Per-source weights used during Reciprocal Rank Fusion.

    Each value scales the contribution of one ranked list to the final
    `hybrid_score`. Set a weight to **0** to disable that source entirely
    (the underlying query will not even be issued). Boost a weight above
    1.0 to bias the ranking toward that signal.
    """
    embedding: float = Field(
        1.0, ge=0, le=10,
        description="Weight for the summary-based vector column (`embedding`).",
    )
    embedding_context: float = Field(
        1.0, ge=0, le=10,
        description=(
            "Weight for the full-context vector column "
            "(`embedding_context`) — best for fact-pattern queries."
        ),
    )
    embedding_ratio: float = Field(
        1.0, ge=0, le=10,
        description=(
            "Weight for the legal-reasoning vector column "
            "(`embedding_ratio`) — best for doctrinal queries."
        ),
    )
    fts: float = Field(
        1.0, ge=0, le=10,
        description=(
            "Weight for the Postgres full-text-search source. Set to 0 on "
            "`/search` to make it semantic-only; ignored on `/search/semantic`."
        ),
    )


class SearchRequest(BaseModel):
    """Request body shared by `POST /search`, `/search/semantic` and `/search/fts`.

    **Query strings** — provide either a single shared `q` or separate
    `q_semantic` / `q_keywords`. Any side that is `null` falls back to `q`.
    On `/search/semantic` only `q_semantic` (or `q`) is required;
    on `/search/fts` only `q_keywords` (or `q`) is required.
    """
    q: Optional[str] = Field(
        None,
        description=(
            "Shared query string used for **both** the semantic and FTS sides "
            "when the side-specific fields are not provided. Most callers "
            "should send only this field."
        ),
        examples=["responsabilidade civil extracontratual do Estado"],
    )
    q_semantic: Optional[str] = Field(
        None,
        description=(
            "Natural-language query embedded for vector search. Use this when "
            "the semantic intent differs from the keywords you'd like FTS to "
            "match (e.g. paraphrase the question here, list legal terms in "
            "`q_keywords`)."
        ),
        examples=["despedimento sem justa causa por uso indevido de email corporativo"],
    )
    q_keywords: Optional[str] = Field(
        None,
        description=(
            "Keyword query for full-text search. Supports "
            "`websearch_to_tsquery` syntax: quoted `\"phrase\"`, `-excluded`, "
            "`OR`. Whitespace acts as AND."
        ),
        examples=["despedimento email corporativo"],
    )
    limit: int = Field(
        20, ge=1, le=100,
        description="Maximum number of results to return (1–100).",
    )
    rrf_k: int = Field(
        50, ge=1,
        description=(
            "Reciprocal Rank Fusion smoothing constant. Lower values make the "
            "top-ranked document of each source dominate; higher values blend "
            "the lists more evenly. Default 50."
        ),
    )
    overfetch: int = Field(
        4, ge=1, le=20,
        description=(
            "Per-source candidates fetched = `limit * overfetch`. Higher "
            "values improve fusion quality (more chance of overlap between "
            "sources) at the cost of latency. Default 4."
        ),
    )
    weights: SearchWeights = Field(
        default_factory=SearchWeights,
        description=(
            "Per-source RRF weights. Defaults give all four sources equal "
            "weight (1.0). Set any to 0 to disable that source."
        ),
    )
    filters: Optional[Filters] = Field(
        None,
        description=(
            "Optional structured filters (court, date range, AUJ-only, …). "
            "See the `Filters` schema or call `GET /filters` for valid values."
        ),
    )


# Reusable, labelled body examples surfaced in Swagger UI's "Try it out" panel.
# Keys are the dropdown labels.
HYBRID_EXAMPLES: dict[str, dict[str, Any]] = {
    "simple": {
        "summary": "Simple hybrid search (single shared string)",
        "description": "Send only `q` and the same string is used for both vectors and FTS.",
        "value": {"q": "responsabilidade civil do Estado", "limit": 10},
    },
    "dual_string_filtered": {
        "summary": "Dual-string + STJ AUJs since 2020",
        "description":
            "Use `q_semantic` for the natural-language intent and "
            "`q_keywords` for the terms FTS should match. Restrict to "
            "binding precedent from the STJ in the last few years.",
        "value": {
            "q_semantic":
                "responsabilidade civil extracontratual do Estado "
                "por funcionamento anormal da justiça",
            "q_keywords": "responsabilidade civil Estado",
            "limit": 10,
            "filters": {
                "court": ["STJ"],
                "is_auj": True,
                "date_from": "2020-01-01",
            },
        },
    },
    "vectors_only_boost_ratio": {
        "summary": "Vectors only (disable FTS), boost the ratio column",
        "description":
            "Disable FTS by zeroing its weight and boost the legal-reasoning "
            "column for a doctrinal query.",
        "value": {
            "q": "interpretação restritiva do conceito de consumidor",
            "limit": 20,
            "weights": {
                "embedding": 1.0,
                "embedding_context": 0.5,
                "embedding_ratio": 1.5,
                "fts": 0,
            },
        },
    },
}

SEMANTIC_EXAMPLES: dict[str, dict[str, Any]] = {
    "simple": {
        "summary": "Plain semantic search",
        "value": {"q": "acidente de trabalho nexo de causalidade", "limit": 10},
    },
    "ratio_only": {
        "summary": "Search only the legal-reasoning column",
        "description": "Useful for purely doctrinal queries.",
        "value": {
            "q": "ónus da prova em matéria de cláusulas contratuais gerais",
            "limit": 10,
            "weights": {
                "embedding": 0,
                "embedding_context": 0,
                "embedding_ratio": 1.0,
            },
        },
    },
}

FTS_EXAMPLES: dict[str, dict[str, Any]] = {
    "simple": {
        "summary": "Plain keyword search",
        "value": {"q": "despedimento sem justa causa", "limit": 10},
    },
    "phrase_and_exclusion": {
        "summary": "Quoted phrase with token exclusion",
        "description":
            "`websearch_to_tsquery` syntax: `\"phrase\"` for adjacency, "
            "`-token` to exclude.",
        "value": {
            "q_keywords": "\"acidente de viação\" -trabalho",
            "limit": 10,
            "filters": {"court": ["STJ", "TRP"]},
        },
    },
}


class SearchResult(BaseModel):
    """A single ranked document in a search response.

    Fields populated depend on which sources contributed to the ranking:

    * `similarity_scores` — present only for the embedding columns that
      returned this document (cosine similarity, in `[0, 1]`, higher = closer).
    * `fts_rank` — Postgres `ts_rank_cd` score, present only when FTS
      matched this document. Not directly comparable to `similarity_scores`.
    * `hybrid_score` — fused RRF score across all enabled sources; this is
      what `results` is sorted by.
    * `source_ranks` — the document's 1-based rank inside each source's
      pre-fusion list (useful for explaining the ranking).
    """
    doc_id: str = Field(
        description="Stable opaque identifier; use with `GET /document/{doc_id}`.",
        examples=["3a8c0d2e9f1b4a7e8d6c5b4a3f2e1d0c"],
    )
    url: str = Field(
        description="Source URL on dgsi.pt.",
        examples=["https://www.dgsi.pt/jstj.nsf/...?OpenDocument"],
    )
    court_short: str = Field(
        description="Court code (`STJ`, `STA`, `TRP`, …); same values used in the `court` filter.",
        examples=["STJ"],
    )
    process_number: Optional[str] = Field(
        None, description="Internal process number assigned by the court.",
        examples=["1234/19.5T8LSB.L1.S1"],
    )
    decision_date: Optional[date] = Field(
        None, description="Date the decision was issued (ISO `YYYY-MM-DD`).",
    )
    legal_domain: Optional[str] = Field(
        None,
        description="Free-text legal domain extracted from the decision (high cardinality).",
    )
    is_auj: Optional[bool] = Field(
        None,
        description=(
            "True only when this decision is itself an Acórdão de "
            "Uniformização / Fixação de Jurisprudência (binding precedent)."
        ),
    )
    summary: Optional[str] = Field(
        None, description="LLM-generated summary (≤ 300 words) of the decision.",
    )
    metadata: Optional[dict] = Field(
        None,
        description=(
            "Full structured extraction (parties, citations, ratio decidendi, "
            "amounts, timeline events, …). Same shape as `extractor.schema."
            "ExtractedInfo`. Use `GET /document/{doc_id}` to fetch this in "
            "isolation."
        ),
    )
    similarity_scores: Optional[dict[str, float]] = Field(
        None,
        description=(
            "Cosine similarity per embedding column that returned this doc. "
            "Keys are a subset of `embedding`, `embedding_context`, "
            "`embedding_ratio`. Range: `[0, 1]`, higher is closer."
        ),
        examples=[{"embedding": 0.83, "embedding_ratio": 0.79}],
    )
    fts_rank: Optional[float] = Field(
        None,
        description=(
            "Postgres `ts_rank_cd` score for the FTS source. Present only "
            "when FTS matched this document."
        ),
    )
    hybrid_score: Optional[float] = Field(
        None,
        description=(
            "Fused Reciprocal Rank Fusion score across all enabled sources. "
            "Higher is better. `results` is sorted by this field."
        ),
    )
    source_ranks: Optional[dict[str, int]] = Field(
        None,
        description=(
            "1-based rank of this document inside each source's pre-fusion "
            "list. Useful to audit which signal pushed the doc to the top."
        ),
        examples=[{"embedding": 2, "embedding_context": 5, "fts": 11}],
    )


class SearchResponse(BaseModel):
    """Wrapper returned by every search endpoint."""
    query_semantic: Optional[str] = Field(
        None,
        description="Effective semantic query used (after `q` fallback). Echoed back for traceability.",
    )
    query_keywords: Optional[str] = Field(
        None,
        description="Effective keyword query used (after `q` fallback).",
    )
    count: int = Field(description="Number of items in `results` (≤ requested `limit`).")
    sources_used: list[str] = Field(
        description=(
            "Sources that actually contributed to the ranking, in the order "
            "they were fused. A subset of "
            "`embedding`, `embedding_context`, `embedding_ratio`, `fts`."
        ),
        examples=[["embedding", "embedding_context", "embedding_ratio", "fts"]],
    )
    mode: str = Field(
        description="Endpoint that produced the response: `hybrid`, `semantic` or `fts`.",
        examples=["hybrid"],
    )
    filters: Optional[Filters] = Field(
        None, description="Filters as received in the request, echoed back.",
    )
    results: list[SearchResult] = Field(
        description="Ranked results, sorted by `hybrid_score` descending.",
    )


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


async def _vector_query(emb_lit: str, field: str, limit: int,
                        filters: Optional[Filters]) -> list[tuple[str, float]]:
    """Single HNSW query against `field`. The embedding literal is reused
    across the 3 columns so we only call the embedding API once."""
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


async def _search_vectors(q: str, fields: list[str], limit: int,
                          filters: Optional[Filters]
                          ) -> dict[str, list[tuple[str, float]]]:
    """Embed `q` once, then run one HNSW query per `fields` entry in parallel.
    Returns {field: [(doc_id, similarity), ...]}."""
    if not fields:
        return {}
    emb = await embed_query(q)
    emb_lit = "[" + ",".join(f"{x:.7f}" for x in emb) + "]"
    results = await asyncio.gather(
        *(_vector_query(emb_lit, f, limit, filters) for f in fields)
    )
    return dict(zip(fields, results))


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


def _rrf_merge_multi(per_source: dict[str, list[tuple[str, float]]],
                     weights: dict[str, float],
                     k: int = 50,
                     ) -> tuple[list[tuple[str, float]], dict[str, dict[str, int]]]:
    """Reciprocal Rank Fusion across an arbitrary number of sources.

    Returns:
      * sorted [(doc_id, fused_score)] (highest score first)
      * {doc_id: {source: 1-based-rank}} for explainability.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for source, hits in per_source.items():
        w = weights.get(source, 0.0)
        if w <= 0:
            continue
        for rank, (doc_id, _) in enumerate(hits, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
            ranks.setdefault(doc_id, {})[source] = rank
    return sorted(scores.items(), key=lambda x: x[1], reverse=True), ranks


@app.get(
    "/",
    tags=["health"],
    summary="Service banner",
    description="Lightweight banner used as a smoke test that the service is reachable.",
)
async def root():
    return {"name": "PT Caselaw DGSI Search API", "status": "ok", "version": "3.0.0"}


@app.get(
    "/health",
    tags=["health"],
    summary="Liveness probe",
    description=(
        "Returns `healthy` when both the database pool and the HTTP client "
        "(used to call the embedding provider) are initialised. Does not "
        "issue any database query, so it is safe to call at high frequency."
    ),
)
async def health():
    ok = db_pool is not None and http_client is not None
    return {"status": "healthy" if ok else "degraded"}


@app.get(
    "/stats",
    tags=["info"],
    summary="Corpus and configuration statistics",
    description=(
        "Returns the total number of documents, how many have each of the "
        "three embedding columns populated, the embedding model in use, and "
        "the HNSW `ef_search` setting. Use this to sanity-check ingestion "
        "coverage before relying on a search variant."
    ),
)
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


def _resolve_queries(req: SearchRequest, need_sem: bool, need_fts: bool
                     ) -> tuple[Optional[str], Optional[str]]:
    """Pick the effective semantic and keyword strings, falling back to `q`.
    Raises 400 if any required slot ends up empty."""
    sem = req.q_semantic if req.q_semantic is not None else req.q
    kw = req.q_keywords if req.q_keywords is not None else req.q
    if need_sem and not sem:
        raise HTTPException(400, "Provide `q` or `q_semantic` for vector search")
    if need_fts and not kw:
        raise HTTPException(400, "Provide `q` or `q_keywords` for keyword search")
    return sem, kw


def _enabled_vector_fields(weights: SearchWeights) -> list[str]:
    return [f for f in EMBEDDING_FIELDS if getattr(weights, f) > 0]


def _build_results(
    merged: list[tuple[str, float]],
    docs: dict[str, dict],
    per_source: dict[str, list[tuple[str, float]]],
    ranks: dict[str, dict[str, int]],
    limit: int,
    include_hybrid: bool,
) -> list[SearchResult]:
    """Assemble SearchResult rows from a merged ranking + per-source maps."""
    sim_maps = {f: dict(per_source[f]) for f in EMBEDDING_FIELDS if f in per_source}
    fts_map = dict(per_source.get("fts", []))
    out: list[SearchResult] = []
    for doc_id, score in merged[:limit]:
        if doc_id not in docs:
            continue
        sim = {f: round(sim_maps[f][doc_id], 4)
               for f in sim_maps if doc_id in sim_maps[f]}
        out.append(SearchResult(
            similarity_scores=sim or None,
            fts_rank=round(fts_map[doc_id], 4) if doc_id in fts_map else None,
            hybrid_score=round(score, 6) if include_hybrid else None,
            source_ranks=ranks.get(doc_id),
            **docs[doc_id],
        ))
    return out


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["search"],
    summary="Hybrid search (3 vectors + FTS, fused with RRF) — recommended",
    description=(
        "Run the query against **all four sources** in parallel and return a "
        "single ranked list:\n\n"
        "1. `embedding` — vector index over the LLM summary.\n"
        "2. `embedding_context` — vector index over the full context "
        "(facts + parties + decision).\n"
        "3. `embedding_ratio` — vector index over the legal reasoning / ratio.\n"
        "4. `fts` — Postgres full-text search (Portuguese + unaccent).\n\n"
        "The query string is embedded **once** (single call to the embedding "
        "provider) and the resulting vector is reused across the 3 HNSW "
        "lookups. Per-source ranks are then combined with weighted "
        "**Reciprocal Rank Fusion**:\n\n"
        "```\nhybrid_score(d) = Σ_source  weight[source] / (rrf_k + rank_source(d))\n```\n\n"
        "Use the `weights` object to disable a source (set to 0) or to bias "
        "the ranking toward a particular signal. Use `filters` to restrict "
        "the candidate set; call `GET /filters` first to discover valid values.\n\n"
        "**400** is returned if every weight is 0, or if the required query "
        "string for an enabled source is missing."
    ),
    responses={
        400: {"description": "Invalid request (no query for an enabled source, or all weights = 0)."},
        502: {"description": "Embedding provider returned a non-200 response."},
    },
)
async def search_hybrid(
    req: SearchRequest = Body(..., openapi_examples=HYBRID_EXAMPLES),
):
    """Hybrid search across all 3 vector columns plus full-text search,
    fused with weighted Reciprocal Rank Fusion. See the OpenAPI description."""
    vec_fields = _enabled_vector_fields(req.weights)
    use_fts = req.weights.fts > 0
    if not vec_fields and not use_fts:
        raise HTTPException(400, "All weights are 0; nothing to search")
    sem_q, kw_q = _resolve_queries(req, need_sem=bool(vec_fields), need_fts=use_fts)
    over = req.limit * req.overfetch

    if vec_fields and use_fts:
        per_vec, fts_hits = await asyncio.gather(
            _search_vectors(sem_q, vec_fields, over, req.filters),
            _search_fts(kw_q, over, req.filters),
        )
    elif vec_fields:
        per_vec = await _search_vectors(sem_q, vec_fields, over, req.filters)
        fts_hits = []
    else:
        per_vec = {}
        fts_hits = await _search_fts(kw_q, over, req.filters)

    per_source: dict[str, list[tuple[str, float]]] = dict(per_vec)
    if use_fts:
        per_source["fts"] = fts_hits
    weights = {f: getattr(req.weights, f) for f in vec_fields}
    if use_fts:
        weights["fts"] = req.weights.fts

    merged, ranks = _rrf_merge_multi(per_source, weights, k=req.rrf_k)
    docs = await _fetch_docs([d for d, _ in merged[:req.limit]])
    results = _build_results(merged, docs, per_source, ranks,
                             req.limit, include_hybrid=True)
    sources_used = [*vec_fields] + (["fts"] if use_fts else [])
    return SearchResponse(query_semantic=sem_q, query_keywords=kw_q,
                          count=len(results), sources_used=sources_used,
                          mode="hybrid", filters=req.filters, results=results)


@app.post(
    "/search/semantic",
    response_model=SearchResponse,
    tags=["search"],
    summary="Semantic-only search (3 vector columns, no FTS)",
    description=(
        "Same as `POST /search` but with full-text search **disabled**. "
        "The `weights.fts` field is ignored. Use this when the query is "
        "purely conceptual / paraphrased and exact-token matching would add "
        "noise (e.g. broad doctrinal questions).\n\n"
        "Only `q` or `q_semantic` is required; `q_keywords` is ignored. "
        "Disable individual vector columns by setting their weight to 0 — "
        "for example, `{\"weights\": {\"embedding_ratio\": 0}}` to ignore "
        "the legal-reasoning column."
    ),
    responses={
        400: {"description": "Missing semantic query, or all vector weights are 0."},
        502: {"description": "Embedding provider returned a non-200 response."},
    },
)
async def search_semantic(
    req: SearchRequest = Body(..., openapi_examples=SEMANTIC_EXAMPLES),
):
    """Semantic-only search across all 3 vector columns in parallel.
    See the OpenAPI description."""
    vec_fields = _enabled_vector_fields(req.weights)
    if not vec_fields:
        raise HTTPException(400, "All vector weights are 0; nothing to search")
    sem_q, _ = _resolve_queries(req, need_sem=True, need_fts=False)
    over = req.limit * req.overfetch

    per_source = await _search_vectors(sem_q, vec_fields, over, req.filters)
    weights = {f: getattr(req.weights, f) for f in vec_fields}
    merged, ranks = _rrf_merge_multi(per_source, weights, k=req.rrf_k)
    docs = await _fetch_docs([d for d, _ in merged[:req.limit]])
    # Single-source results don't really have a "hybrid" score, but multiple
    # vector columns are still being fused — keep it for transparency.
    include_hybrid = len(vec_fields) > 1
    results = _build_results(merged, docs, per_source, ranks,
                             req.limit, include_hybrid=include_hybrid)
    return SearchResponse(query_semantic=sem_q, count=len(results),
                          sources_used=vec_fields, mode="semantic",
                          filters=req.filters, results=results)


@app.post(
    "/search/fts",
    response_model=SearchResponse,
    tags=["search"],
    summary="Full-text-only search (Postgres GIN, Portuguese + unaccent)",
    description=(
        "Pure keyword search via Postgres `to_tsvector('portuguese', …)` + "
        "`unaccent`. Supports the standard `websearch_to_tsquery` syntax:\n\n"
        "* whitespace = AND  (`despedimento email`)\n"
        "* `OR`              (`despedimento OR demissão`)\n"
        "* `\"phrase\"`     for adjacency\n"
        "* `-token`        to exclude a token\n\n"
        "Only `q` or `q_keywords` is required; `q_semantic` and the vector "
        "weights are ignored. No embedding API call is performed, so this "
        "is the cheapest and lowest-latency endpoint — ideal for exact "
        "lookups (process numbers, judge names, statute references)."
    ),
    responses={
        400: {"description": "Missing keyword query, or `weights.fts` is 0."},
    },
)
async def search_fts(
    req: SearchRequest = Body(..., openapi_examples=FTS_EXAMPLES),
):
    """Full-text-only search via the GIN `fts` index. See the OpenAPI description."""
    if req.weights.fts <= 0:
        raise HTTPException(400, "weights.fts is 0; nothing to search")
    _, kw_q = _resolve_queries(req, need_sem=False, need_fts=True)
    over = req.limit * req.overfetch

    fts_hits = await _search_fts(kw_q, over, req.filters)
    per_source = {"fts": fts_hits}
    weights = {"fts": req.weights.fts}
    merged, ranks = _rrf_merge_multi(per_source, weights, k=req.rrf_k)
    docs = await _fetch_docs([d for d, _ in merged[:req.limit]])
    results = _build_results(merged, docs, per_source, ranks,
                             req.limit, include_hybrid=False)
    return SearchResponse(query_keywords=kw_q, count=len(results),
                          sources_used=["fts"], mode="fts",
                          filters=req.filters, results=results)


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


@app.get(
    "/filters",
    tags=["info"],
    summary="Discover supported filter fields and their available values",
    description=(
        "Returns, for each field accepted by the `Filters` object on the "
        "search endpoints:\n\n"
        "* `court` — every `court_short` present in the corpus, with counts.\n"
        "* `is_auj` — distribution of true/false.\n"
        "* `decision_date` — `min` / `max` / `count`.\n"
        "* `legal_domain` — `distinct_count` and the top-N most frequent "
        "values. This field has 6000+ unique values, so use the "
        "`legal_domain_prefix` query parameter to autocomplete.\n"
        "* `decision_type`, `extraction_confidence` — full enumerations.\n\n"
        "**Caching** — the heavy aggregations are cached in-memory for "
        "`FILTERS_CACHE_TTL` seconds (default 3600). The response always "
        "echoes `cached` (bool) and `cache_age_seconds`. Pass `refresh=true` "
        "to bypass the cache.\n\n"
        "**Recommended workflow** — call this endpoint once at the start of "
        "an agent session to ground every later `filters` you send, then "
        "send only values that appear here."
    ),
)
async def get_filters(
    legal_domain_prefix: Optional[str] = Query(
        None,
        description=(
            "Case-insensitive substring used to filter the `legal_domain` "
            "enumeration before truncating to `top_legal_domains`. Useful "
            "for autocomplete-style discovery."
        ),
        examples=["insolvencia"],
    ),
    top_legal_domains: int = Query(
        50, ge=1, le=500,
        description=(
            "How many `legal_domain` values to return (most frequent first). "
            "Capped at 500 because the field is high-cardinality."
        ),
    ),
    refresh: bool = Query(
        False,
        description="Bypass the in-memory cache and recompute the aggregations.",
    ),
):
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


@app.get(
    "/document/{doc_id}",
    tags=["documents"],
    summary="Fetch the full record for a single document",
    description=(
        "Returns the same fields exposed in `SearchResult` for the given "
        "`doc_id`, including the full `metadata` JSON (parties, citations, "
        "ratio decidendi, amounts, timeline events, …; same shape as "
        "`extractor.schema.ExtractedInfo`). Returns **404** if the id is "
        "unknown.\n\n"
        "Typical usage: after a `/search` call, pick the `doc_id` of an "
        "interesting result and hydrate it here for full context."
    ),
    responses={
        404: {"description": "No document found for the given `doc_id`."},
    },
)
async def get_document(
    doc_id: str = FastAPIPath(  # noqa: F821 — imported just below
        ...,
        description="Stable opaque document identifier returned by any /search endpoint.",
        examples=["3a8c0d2e9f1b4a7e8d6c5b4a3f2e1d0c"],
    ),
):
    docs = await _fetch_docs([doc_id])
    if doc_id not in docs:
        raise HTTPException(404, "Document not found")
    return docs[doc_id]
