"""FastAPI app for hybrid search over DGSI Portuguese caselaw.

Uses Supabase Postgres directly via asyncpg:
  * HNSW indexes on `embedding` / `embedding_context` / `embedding_ratio`
    (halfvec_cosine_ops) for vector search.
  * GIN index on `fts` (tsvector built with unaccent + portuguese) for FTS.
  * B-tree expression indexes on `metadata->>'decision_type'` and
    `metadata->>'extraction_confidence'` so the planner has real selectivity
    statistics for jsonb filters (without them, the 0.5% default estimate
    made filtered vector queries take the exact-sort path — 30s+ timeouts).
"""
import asyncio
import json as _json
import logging
import os
import re
import secrets
import time
import unicodedata
from contextlib import asynccontextmanager
from datetime import date

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Any, Literal, Optional, Union

import asyncpg
import httpx
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Path as FastAPIPath, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from api.judges import judge_key, query_tokens
from api.legal_areas import CANONICAL_AREAS, resolve_legal_areas

# Use absolute path so the server works regardless of the working directory
# (e.g. when launched as a Claude Desktop subprocess from the home directory).
_ENV_FILE = Path(__file__).parent.parent / ".env.local"
load_dotenv(_ENV_FILE)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
# Must match the model that produced the stored vectors (verified 2026-10).
EMBEDDING_MODEL = "google/gemini-embedding-001"
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "4"))
# Fast model for short legislation-query canonicalization/selection calls.
LLM_MODEL = "google/gemini-3.1-flash-lite"
# Tried in order. OpenRouter's /rerank has no `models` fallback list and Voyage
# has a single provider there, so the fallback lives here (issue #8: Voyage
# returned 429 for every request from 2026-10-10).
RERANK_MODELS: tuple[str, ...] = ("voyageai/rerank-2.5", "cohere/rerank-v3.5")
RERANK_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
RERANK_MAX_ATTEMPTS = 3  # per model
RERANK_BACKOFF_BASE_SECONDS = 0.5  # 0.5s, 1s between attempts
RERANK_MAX_RETRY_DELAY_SECONDS = 2.0
# A model still answering 429 after its retries is skipped for this long (or
# its Retry-After, up to the max) so requests go straight to the next model.
RERANK_COOLDOWN_SECONDS = 60.0
RERANK_MAX_COOLDOWN_SECONDS = 600.0
# Reranker defaults chosen with scripts/evaluate_search_quality.py (2026-10):
# letting Voyage decide the order (weight 20 vs RRF's ~1/(k+rank) per source)
# over the top 150 fused candidates raised hybrid P@5 0.677 -> 0.746.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "150"))
RERANK_WEIGHT = float(os.getenv("RERANK_WEIGHT", "20.0"))
# When True, reranked documents that were neither in the fused top 5 nor FTS
# hits are kept out of the top 5. Off by default: it blocked the reranker from
# promoting strong vector-only matches (17 cases improved, 8 lost one result).
RERANK_LEXICAL_GUARD = False
RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "30"))
LEGISLATION_RERANK_WEIGHT = float(os.getenv("LEGISLATION_RERANK_WEIGHT", "4.0"))
# gemini-3.1-pro-preview gave identical results on the semantic legislation
# benchmark cases but took ~16s p50 vs ~3.3s (2026-10).
LEGISLATION_RESOLVER_MODEL = "google/gemini-3.1-flash-lite"

DB_HOST = os.getenv("SUPABASE_DB_HOST", "")
DB_PORT = int(os.getenv("SUPABASE_DB_PORT", "5432"))
DB_USER = os.getenv("SUPABASE_DB_USER", "")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")

# HNSW search-time recall/speed tradeoff. 40 is a good default; raise for recall.
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "40"))
# Iterative scan mode for filtered HNSW queries (pgvector >= 0.8).
# "strict_order" keeps walking the graph until LIMIT rows match the filters —
# exact ordering but UNBOUNDED: an anti-correlated filter/query combination can
# walk the whole graph and hit the 30s statement timeout. "relaxed_order" stops
# at hnsw.max_scan_tuples, trading a small chance of fewer-than-LIMIT results
# for a hard latency bound — the right posture for a client-facing API.
_HNSW_ITERATIVE_MODES = ("off", "strict_order", "relaxed_order")
HNSW_ITERATIVE_SCAN = os.getenv("HNSW_ITERATIVE_SCAN", "relaxed_order")
if HNSW_ITERATIVE_SCAN not in _HNSW_ITERATIVE_MODES:
    raise ValueError(
        f"HNSW_ITERATIVE_SCAN must be one of {_HNSW_ITERATIVE_MODES}, "
        f"got {HNSW_ITERATIVE_SCAN!r}"
    )
HNSW_MAX_SCAN_TUPLES = int(os.getenv("HNSW_MAX_SCAN_TUPLES", "20000"))

EMBEDDING_FIELDS: tuple[str, ...] = ("embedding", "embedding_context", "embedding_ratio")
ALL_SOURCES: tuple[str, ...] = (*EMBEDDING_FIELDS, "fts")

DOC_COLUMN_NAMES = (
    "doc_id", "url", "court_short", "process_number", "decision_date",
    "legal_domain", "is_auj", "summary", "metadata",
)
DOC_COLUMNS = ", ".join(DOC_COLUMN_NAMES)
LEGISLATION_CITATIONS_TABLE = "document_legislation_citations"


def _qualified_doc_columns(alias: str) -> str:
    """Return document columns qualified with a SQL alias."""
    return ", ".join(f"{alias}.{col}" for col in DOC_COLUMN_NAMES)

# Initialised at startup
db_pool: Optional[asyncpg.Pool] = None
http_client: Optional[httpx.AsyncClient] = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Per-connection setup: tune HNSW ef_search and iterative scan so
    filtered semantic queries stay fast and latency-bounded."""
    await conn.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}")
    # pgvector >= 0.8: keep walking the HNSW graph when WHERE filters
    # would otherwise leave the LIMIT under-filled — but cap the walk so a
    # pathological filter/query combo degrades to fewer results instead of
    # a full-graph scan that blows the 30s statement timeout.
    # (HNSW_ITERATIVE_SCAN is validated against a whitelist at import time.)
    await conn.execute(f"SET hnsw.iterative_scan = {HNSW_ITERATIVE_SCAN}")
    if HNSW_ITERATIVE_SCAN != "off":
        await conn.execute(f"SET hnsw.max_scan_tuples = {HNSW_MAX_SCAN_TUPLES}")


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
    # Probe the embedding API so a missing/invalid key is caught at startup.
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set — embedding search will not work.")
    yield
    await db_pool.close()
    await http_client.aclose()


API_DESCRIPTION = """
Hybrid search over Portuguese court decisions (DGSI: STJ, STA, TR*, TCA*, TC, JP, …).

## How to use this API (for LLM agents)

The intended workflow is:

1. **Discover what you can filter on** with `GET /filters`. This returns the
   available `court_short` codes with document counts, the corpus date range,
   and AUJ vs non-AUJ counts. Use this to ground any filter you later send.
2. **Search** with `POST /search` — it fuses 3 vector representations and
   full-text search into one ranking. To find decisions that cite specific
   legislation instead, use `POST /search/legislation` (exact article
   references) or `POST /search/legislation/semantic` (free-form law
   queries with LLM reranking).
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

`SearchRequest` accepts two query inputs:

* `q_semantic` — natural-language description of what you're looking for
  (e.g. *"despedimento sem justa causa por uso indevido de email corporativo"*).
  This is what gets embedded.
* `q_keywords` — list of keywords for FTS (e.g. `["despedimento", "email",
  "corporativo"]`), matched with AND logic. Each entry supports
  `websearch_to_tsquery` syntax (quoted phrases, `-exclusion`, `OR`).

There is no shared fallback field: under the default weights both vector
search and FTS are enabled, so `q_semantic` and `q_keywords` are each
required for their respective sources (a missing one returns 400). If you
only have one string, send it as `q_semantic` and disable FTS with
`{"weights": {"fts": 0}}` — or split it yourself: full phrasing in
`q_semantic`, key terms in `q_keywords`.

## Filters

All search endpoints accept the same `filters` object (AND across fields).
For convenience, the four most common filters can also be passed as
top-level fields directly on the request body — no nesting required:

| Top-level field | Equivalent `filters` field | Type |
|-----------------|---------------------------|------|
| `courts`        | `filters.court`            | `list[str]` — e.g. `["STJ", "TRP"]` |
| `from_date`     | `filters.date_from`        | date string (see formats below) |
| `to_date`       | `filters.date_to`          | date string (see formats below) |
| `is_auj`        | `filters.is_auj`           | `bool` |

When a top-level field and its `filters` counterpart are both present,
the top-level field wins.

### Date formats

Date fields (`from_date`, `to_date`, `filters.date_from`, `filters.date_to`)
accept any of the following formats — the API normalises them automatically:

| Format         | Example        |
|----------------|----------------|
| `YYYY-MM-DD`   | `2024-01-31`   |
| `YYYY/MM/DD`   | `2024/01/31`   |
| `YYYY.MM.DD`   | `2024.01.31`   |
| `DD-MM-YYYY`   | `31-01-2024`   |
| `DD/MM/YYYY`   | `31/01/2024`   |
| `DD.MM.YYYY`   | `31.01.2024`   |

ISO 8601 (`YYYY-MM-DD`) is the recommended format; the others are accepted
to accommodate data from Portuguese legal databases and spreadsheets that
commonly use day-first or slash-separated dates.

See the `Filters` schema for every supported field.
`GET /filters` returns the available court codes and the corpus date range.

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
        "Search over the corpus: `POST /search` (hybrid vector + keyword, "
        "body: `SearchRequest`) and the `/search/legislation*` endpoints "
        "(decisions citing given statutes, their own request schemas)."},
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

# ---------------------------------------------------------------------------
# OAuth 2.0 endpoints (open / no real auth — exists so MCP clients can
# complete the discovery + token flow and connect without errors)
# ---------------------------------------------------------------------------

@app.get("/.well-known/oauth-authorization-server", include_in_schema=False)
async def oauth_authorization_server_metadata(request: Request):
    base = f"{request.url.scheme}://{request.url.netloc}"
    return {
        "issuer": base,
        "authorization_endpoint": f"{base}/authorize",
        "token_endpoint": f"{base}/token",
        "registration_endpoint": f"{base}/register",
        "scopes_supported": ["mcp"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "client_credentials"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_basic", "client_secret_post"],
    }


@app.post("/register", include_in_schema=False)
async def oauth_register(request: Request):
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass
    return {
        "client_id": secrets.token_urlsafe(16),
        "client_secret": secrets.token_urlsafe(32),
        "client_id_issued_at": int(time.time()),
        "client_secret_expires_at": 0,
        "redirect_uris": body.get("redirect_uris", []),
        "grant_types": ["authorization_code", "client_credentials"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }


@app.get("/authorize", include_in_schema=False)
async def oauth_authorize(
    redirect_uri: str,
    state: str = "",
    code_challenge: str = "",
    code_challenge_method: str = "S256",
    client_id: str = "",
    response_type: str = "code",
):
    """Immediately redirect back with an auth code — no login required."""
    code = secrets.token_urlsafe(32)
    url = f"{redirect_uri}?code={code}"
    if state:
        url += f"&state={state}"
    return RedirectResponse(url=url, status_code=302)


@app.post("/token", include_in_schema=False)
async def oauth_token(request: Request):
    """Issue a Bearer token to anyone who asks."""
    return {
        "access_token": secrets.token_urlsafe(32),
        "token_type": "Bearer",
        "expires_in": 86400,
        "scope": "mcp",
    }


_DATE_FORMATS = (
    "%Y-%m-%d",   # 2024-01-31  (ISO — checked first)
    "%Y/%m/%d",   # 2024/01/31
    "%Y.%m.%d",   # 2024.01.31
    "%d-%m-%Y",   # 31-01-2024
    "%d/%m/%Y",   # 31/01/2024
    "%d.%m.%Y",   # 31.01.2024
)


def _parse_flexible_date(v: object) -> object:
    """Coerce a date-like string into a ``datetime.date``.

    Accepts ISO 8601 (``YYYY-MM-DD``) and the common Portuguese / European
    variants that use ``/`` or ``.`` as separators, or day-first order:

    * ``YYYY-MM-DD``, ``YYYY/MM/DD``, ``YYYY.MM.DD``
    * ``DD-MM-YYYY``, ``DD/MM/YYYY``, ``DD.MM.YYYY``

    Already-parsed ``date`` objects are passed through unchanged.
    Raises ``ValueError`` on unrecognised input so Pydantic surfaces a clear
    422 validation error to the caller.
    """
    if v is None or isinstance(v, date):
        return v
    if not isinstance(v, str):
        return v  # let Pydantic handle non-string oddities
    s = v.strip()
    for fmt in _DATE_FORMATS:
        try:
            from datetime import datetime
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    accepted = ", ".join(f"``{f}``" for f in ("YYYY-MM-DD", "YYYY/MM/DD", "DD-MM-YYYY", "DD/MM/YYYY"))
    raise ValueError(
        f"Unrecognised date format {s!r}. Accepted formats: {accepted}"
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
            "Legal area filter, accent- and case-insensitive. The value is "
            "resolved to canonical areas (see `legal_areas` in `GET /filters`), "
            "including synonyms — `fiscal` → Direito Tributário, `criminal` → "
            "Direito Penal + Processual Penal, `penal` → both penal areas. "
            "Values that match no canonical area fall back to a substring "
            "match on the raw extracted domain."
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
        description=(
            "Inclusive lower bound on `decision_date`. "
            "Accepts `YYYY-MM-DD` (ISO), `YYYY/MM/DD`, `YYYY.MM.DD`, "
            "`DD-MM-YYYY`, `DD/MM/YYYY`, or `DD.MM.YYYY`."
        ),
        examples=["2020-01-01"],
    )
    date_to: Optional[date] = Field(
        None,
        description=(
            "Inclusive upper bound on `decision_date`. "
            "Accepts `YYYY-MM-DD` (ISO), `YYYY/MM/DD`, `YYYY.MM.DD`, "
            "`DD-MM-YYYY`, `DD/MM/YYYY`, or `DD.MM.YYYY`."
        ),
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
    disability_points_gt: Optional[float] = Field(
        None, ge=0, le=100,
        description=(
            "Keep decisions whose highest permanent disability degree "
            "(défice funcional permanente points or IPP %, 0–100, parsed from "
            "the extracted injuries) is strictly greater than this value. "
            "Decisions with no stated degree are excluded."
        ),
        examples=[15],
    )
    disability_points_gte: Optional[float] = Field(
        None, ge=0, le=100,
        description="Like `disability_points_gt`, inclusive.",
    )
    disability_points_lt: Optional[float] = Field(
        None, ge=0, le=100,
        description="Upper bound on the disability degree, exclusive.",
    )
    disability_points_lte: Optional[float] = Field(
        None, ge=0, le=100,
        description="Upper bound on the disability degree, inclusive.",
        examples=[30],
    )
    judge: Optional[list[str]] = Field(
        None, max_length=20,
        description=(
            "Restrict to decisions whose judge rapporteur (relator) is one of "
            "these names (ANY-of). Matching ignores case, accents and "
            "punctuation, so `José Correia` also finds `JOSE CORREIA`; names "
            "must otherwise be complete. Use `GET /judges` to find the names "
            "present in the corpus."
        ),
        examples=[["GARCIA CALEJO"]],
    )

    @field_validator("date_from", "date_to", mode="before")
    @classmethod
    def _coerce_date(cls, v: object) -> object:
        return _parse_flexible_date(v)

    @field_validator("judge")
    @classmethod
    def _judge_keys(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return None
        keys = list(dict.fromkeys(k for k in (judge_key(name) for name in v) if k))
        return keys or None


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
            "Weight for the Postgres full-text-search source. Set to 0 to "
            "disable FTS and run a purely semantic search."
        ),
    )


class SemanticFieldQueries(BaseModel):
    """Optional per-column texts for vector search.

    Each stored vector is built from different LLM-written fields of a
    decision (`embedding`: the one-sentence case description;
    `embedding_context`: that sentence + legal question + summary;
    `embedding_ratio`: ratio decidendi + outcome). Callers that write the same
    fields for the query can match them column to column. Columns left empty
    fall back to `q_semantic`.
    """
    embedding: Optional[str] = Field(
        None, description="Text embedded for the `embedding` column."
    )
    embedding_context: Optional[str] = Field(
        None, description="Text embedded for the `embedding_context` column."
    )
    embedding_ratio: Optional[str] = Field(
        None, description="Text embedded for the `embedding_ratio` column."
    )


class SearchRequest(BaseModel):
    """Request body for `POST /search` (hybrid search).

    **Query strings** — provide `q_semantic` for vector search and/or
    `q_keywords` for full-text search. Each is used exclusively for its
    respective source; there is no shared fallback between the two.

    **Filters** — common filters can be provided as top-level fields
    (`courts`, `from_date`, `to_date`, `is_auj`) for convenience, or as a
    nested `filters` object. Top-level fields take precedence over the
    corresponding fields inside `filters` when both are supplied.
    """
    q_semantic: Optional[str] = Field(
        None,
        description=(
            "Natural-language query embedded for vector search. Required when "
            "any vector weight is non-zero. Phrase the semantic intent here "
            "(e.g. a question or paraphrase); list matching terms in "
            "`q_keywords` for the FTS side."
        ),
        examples=["despedimento sem justa causa por uso indevido de email corporativo"],
    )
    q_keywords: Optional[list[str]] = Field(
        None,
        description=(
            "List of keywords for full-text search. Required when `weights.fts` "
            "is non-zero. Each keyword is matched with AND logic; individual "
            "entries support `websearch_to_tsquery` syntax: quoted `\"phrase\"`, "
            "`-excluded`, `OR`."
        ),
        examples=[["despedimento", "email", "corporativo"]],
    )
    q_semantic_fields: Optional[SemanticFieldQueries] = Field(
        None,
        description=(
            "Optional per-column texts for vector search, written like the "
            "stored fields of a decision. Each enabled vector column embeds "
            "its own text, falling back to `q_semantic`. The reranker still "
            "ranks against `q_semantic`, so keep the user's question there."
        ),
    )
    rerank_mode: Literal["blend", "question"] = Field(
        "blend",
        description=(
            "`blend` (default) adds the reranker's rank to the fused score and "
            "gives it the keywords too. `question` reranks against `q_semantic` "
            "alone and returns the reranker's order, which ranked relevant "
            "decisions higher for long natural-language requests."
        ),
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
            "See the `Filters` schema or call `GET /filters` for valid values. "
            "Top-level shorthand fields (`courts`, `from_date`, `to_date`, "
            "`is_auj`) override the corresponding fields here when both are set."
        ),
    )
    # Convenience top-level filter shorthands
    courts: Optional[list[str]] = Field(
        None,
        description=(
            "Shorthand for `filters.court`. Restrict to one or more courts "
            "using `court_short` codes (ANY-of). Takes precedence over "
            "`filters.court` when both are provided."
        ),
        examples=[["STJ", "TRP"]],
    )
    from_date: Optional[date] = Field(
        None,
        description=(
            "Shorthand for `filters.date_from`. Inclusive lower bound on "
            "`decision_date`. Takes precedence over `filters.date_from` when "
            "both are provided. "
            "Accepts `YYYY-MM-DD` (ISO), `YYYY/MM/DD`, `YYYY.MM.DD`, "
            "`DD-MM-YYYY`, `DD/MM/YYYY`, or `DD.MM.YYYY`."
        ),
        examples=["2020-01-01"],
    )
    to_date: Optional[date] = Field(
        None,
        description=(
            "Shorthand for `filters.date_to`. Inclusive upper bound on "
            "`decision_date`. Takes precedence over `filters.date_to` when "
            "both are provided. "
            "Accepts `YYYY-MM-DD` (ISO), `YYYY/MM/DD`, `YYYY.MM.DD`, "
            "`DD-MM-YYYY`, `DD/MM/YYYY`, or `DD.MM.YYYY`."
        ),
        examples=["2024-12-31"],
    )
    is_auj: Optional[bool] = Field(
        None,
        description=(
            "Shorthand for `filters.is_auj`. `true` keeps only AUJs, `false` "
            "excludes them, `null`/omit returns both. Takes precedence over "
            "`filters.is_auj` when both are provided."
        ),
        examples=[True],
    )

    @field_validator("from_date", "to_date", mode="before")
    @classmethod
    def _coerce_date(cls, v: object) -> object:
        return _parse_flexible_date(v)

    def resolved_filters(self) -> Optional[Filters]:
        """Return the effective Filters, merging top-level shorthands over
        the nested `filters` object. Returns None when nothing is set."""
        base = self.filters or Filters()
        merged = base.model_copy(update={
            "court": self.courts if self.courts is not None else base.court,
            "date_from": self.from_date if self.from_date is not None else base.date_from,
            "date_to": self.to_date if self.to_date is not None else base.date_to,
            "is_auj": self.is_auj if self.is_auj is not None else base.is_auj,
        })
        if all(v is None for v in merged.model_dump().values()):
            return None
        return merged


# Reusable, labelled body examples surfaced in Swagger UI's "Try it out" panel.
# Keys are the dropdown labels.
HYBRID_EXAMPLES: dict[str, dict[str, Any]] = {
    "hybrid": {
        "summary": "Hybrid search (semantic + keyword)",
        "description":
            "Provide `q_semantic` for the natural-language intent and "
            "`q_keywords` for the terms FTS should match.",
        "value": {
            "q_semantic": "responsabilidade civil extracontratual do Estado",
            "q_keywords": ["responsabilidade", "civil", "Estado"],
            "limit": 10,
        },
    },
    "hybrid_field_matched": {
        "summary": "Hybrid search matched field to field",
        "description":
            "Keep the user's question in `q_semantic` (the reranker reads it), "
            "write per-column texts in the style of the stored fields, and "
            "pass keyword phrases joined with `OR` for FTS.",
        "value": {
            "q_semantic":
                "O Estado responde pelos danos causados pela demora excessiva "
                "de um processo judicial?",
            "q_semantic_fields": {
                "embedding":
                    "Ação de indemnização contra o Estado por atraso na decisão "
                    "de processo judicial, discutindo a violação do direito a "
                    "decisão em prazo razoável.",
                "embedding_context":
                    "Ação de indemnização contra o Estado por atraso na decisão "
                    "de processo judicial.\n\nO atraso na decisão constitui "
                    "funcionamento anormal da justiça que obriga o Estado a indemnizar?",
                "embedding_ratio":
                    "O Estado responde civilmente pelos danos decorrentes da "
                    "violação do direito a uma decisão em prazo razoável.",
            },
            "q_keywords": [
                '"responsabilidade civil do estado" OR "atraso na justica" '
                'OR "prazo razoavel"'
            ],
            "limit": 10,
        },
    },
    "hybrid_filtered": {
        "summary": "Hybrid search + STJ AUJs since 2020 (nested filters)",
        "description":
            "Same dual-query approach restricted to binding precedent "
            "from the STJ in the last few years, using the nested `filters` object.",
        "value": {
            "q_semantic":
                "responsabilidade civil extracontratual do Estado "
                "por funcionamento anormal da justiça",
            "q_keywords": ["responsabilidade", "civil", "Estado"],
            "limit": 10,
            "filters": {
                "court": ["STJ"],
                "is_auj": True,
                "date_from": "2020-01-01",
            },
        },
    },
    "hybrid_filtered_shorthand": {
        "summary": "Hybrid search + STJ AUJs since 2020 (top-level shorthands)",
        "description":
            "Identical query using the convenience top-level fields "
            "`courts`, `is_auj`, and `from_date` instead of a nested "
            "`filters` object. Date formats with `/` or `.` separators "
            "and day-first order are also accepted (e.g. `01/01/2020`).",
        "value": {
            "q_semantic":
                "responsabilidade civil extracontratual do Estado "
                "por funcionamento anormal da justiça",
            "q_keywords": ["responsabilidade", "civil", "Estado"],
            "limit": 10,
            "courts": ["STJ"],
            "is_auj": True,
            "from_date": "2020-01-01",
        },
    },
    "vectors_only_boost_ratio": {
        "summary": "Vectors only (disable FTS), boost the ratio column",
        "description":
            "Disable FTS by zeroing its weight and boost the legal-reasoning "
            "column for a doctrinal query.",
        "value": {
            "q_semantic": "interpretação restritiva do conceito de consumidor",
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
        description="Search mode that produced the response: `hybrid`, or `browse` for `POST /browse`.",
        examples=["hybrid"],
    )
    filters: Optional[Filters] = Field(
        None, description="Filters as received in the request, echoed back.",
    )
    results: list[SearchResult] = Field(
        description="Ranked results, sorted by `hybrid_score` descending.",
    )


def _legal_domain_clause(column: str, value: str, idx: int) -> tuple[str, Any]:
    """Filter on canonical legal areas (accent/synonym-insensitive, via
    legal_domain_map); fall back to an unaccented substring match when the
    input resolves to no canonical area."""
    areas = resolve_legal_areas(value)
    if areas:
        return (
            f"{column} IN (SELECT raw FROM legal_domain_map WHERE areas && ${idx}::text[])",
            areas,
        )
    return f"unaccent({column}) ILIKE unaccent(${idx})", f"%{value}%"


# Non-canonical copies of decisions DGSI published more than once
# (scripts/mark_duplicates.py) are never returned by search.
NOT_DUPLICATE_DOC_IDS = (
    "NOT IN (SELECT doc_id FROM documents WHERE duplicate_of IS NOT NULL)"
)


def _build_filters(
    f: Optional[Filters], start_idx: int = 1, alias: str = ""
) -> tuple[str, list[Any]]:
    """Build a SQL WHERE fragment (without 'WHERE'/'AND' prefix) and the
    matching positional params list. The first placeholder will be `$start_idx`.
    Always excludes documents marked as duplicates.
    """
    prefix = f"{alias}." if alias else ""
    clauses: list[str] = [f"{prefix}duplicate_of IS NULL"]
    params: list[Any] = []
    if f is None:
        return clauses[0], params
    idx = start_idx
    if f.court:
        clauses.append(f"{prefix}court_short = ANY(${idx}::text[])")
        params.append(f.court)
        idx += 1
    if f.legal_domain:
        clause, param = _legal_domain_clause(f"{prefix}legal_domain", f.legal_domain, idx)
        clauses.append(clause)
        params.append(param)
        idx += 1
    if f.is_auj is not None:
        clauses.append(f"{prefix}is_auj = ${idx}")
        params.append(f.is_auj)
        idx += 1
    if f.date_from is not None:
        clauses.append(f"{prefix}decision_date >= ${idx}")
        params.append(f.date_from)
        idx += 1
    if f.date_to is not None:
        clauses.append(f"{prefix}decision_date <= ${idx}")
        params.append(f.date_to)
        idx += 1
    if f.decision_type:
        clauses.append(f"{prefix}metadata->>'decision_type' = ANY(${idx}::text[])")
        params.append(f.decision_type)
        idx += 1
    if f.extraction_confidence:
        clauses.append(f"{prefix}metadata->>'extraction_confidence' = ANY(${idx}::text[])")
        params.append(f.extraction_confidence)
        idx += 1
    clause, points_params, idx = _disability_points_clause(f, idx, prefix)
    if clause:
        clauses.append(clause)
        params.extend(points_params)
    clause, judge_params, idx = _judge_clause(f, idx, prefix)
    if clause:
        clauses.append(clause)
        params.extend(judge_params)
    return " AND ".join(clauses), params


def _disability_points_clause(
    f: Filters, idx: int, prefix: str
) -> tuple[str, list[Any], int]:
    """`disability_points_*` bounds as a semi-join on document_disability_points
    (migrations/20261015_document_disability_points.sql)."""
    conditions: list[str] = []
    params: list[Any] = []
    for op, value in (
        (">", f.disability_points_gt), (">=", f.disability_points_gte),
        ("<", f.disability_points_lt), ("<=", f.disability_points_lte),
    ):
        if value is not None:
            conditions.append(f"points {op} ${idx}")
            params.append(value)
            idx += 1
    if not conditions:
        return "", [], idx
    return (
        f"{prefix}doc_id IN (SELECT doc_id FROM document_disability_points "
        f"WHERE {' AND '.join(conditions)})",
        params,
        idx,
    )


def _judge_clause(
    f: Filters, idx: int, prefix: str
) -> tuple[str, list[Any], int]:
    """`judge` as a semi-join on document_judges
    (migrations/20261021_document_judges.sql). Filters normalises the names
    to judge_key."""
    if not f.judge:
        return "", [], idx
    return (
        f"{prefix}doc_id IN (SELECT doc_id FROM document_judges "
        f"WHERE judge_key = ANY(${idx}::text[]))",
        [f.judge],
        idx + 1,
    )


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed several texts in one request (results in input order)."""
    payload = {"model": EMBEDDING_MODEL, "input": texts, "dimensions": EMBEDDING_DIM}
    logger.debug("embed_texts: model=%s dim=%s n=%d", EMBEDDING_MODEL, EMBEDDING_DIM, len(texts))
    resp = None
    for attempt in range(EMBEDDING_MAX_RETRIES + 1):
        try:
            resp = await http_client.post(
                f"{OPENROUTER_BASE}/embeddings",
                json=payload,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            )
        except httpx.RequestError as exc:
            if attempt == EMBEDDING_MAX_RETRIES:
                raise HTTPException(502, f"Embedding API network error: {exc}") from exc
            await asyncio.sleep(min(2.0 * (2**attempt), 8.0))
            continue
        if (
            resp.status_code not in {429, 500, 502, 503, 504}
            or attempt == EMBEDDING_MAX_RETRIES
        ):
            break
        retry_after = resp.headers.get("Retry-After")
        delay = min(float(retry_after), 8.0) if retry_after else min(2.0 * (2**attempt), 8.0)
        await asyncio.sleep(delay)
    assert resp is not None
    body = resp.json()
    if resp.status_code != 200:
        logger.error("embed_query: HTTP %d — %s", resp.status_code, resp.text[:500])
        raise HTTPException(502, f"Embedding API error: {resp.status_code} {resp.text[:200]}")
    # Some providers return 200 with a top-level "error" key instead of "data"
    # (e.g. missing/invalid API key, quota exceeded, unsupported model).
    if "data" not in body:
        logger.error(
            "embed_query: HTTP 200 but no 'data' key. top-level keys=%s body=%s",
            list(body.keys()),
            str(body)[:500],
        )
        err = body.get("error", body)
        raise HTTPException(502, f"Embedding API returned 200 but no 'data' key: {err}")
    data = sorted(body["data"], key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in data]


_LEGISLATION_NORMALIZE_PROMPT = """You are a Portuguese legal citation normalizer.
Given a user's messy or informal reference to a Portuguese law, output ONLY a clean canonical citation string.

Rules:
- Use standard abbreviations when possible: CT, CPC, CC, CP, CPP, CRP, CPA, CPTA, CSC, CCP, RCP, CE, CmC, CRC, CNot, LGT, CEP, CJM
- For ordinary statutes use: "Lei n.º X/YYYY" or "DL YYYY"
- Article format (only when an article IS mentioned): "art. N.º" or "art. N.º, n.º M" or "art. N.º, n.º M, alínea X)"
- If the input does NOT mention a specific article, output ONLY the law abbreviation/name, with NO article.

Examples:
- "artigo 2 co codigo do trabalho" → "CT art. 2"
- "código civil art 500" → "CC art. 500"
- "artigo 394 do codigo do trabalho n 2" → "CT art. 394, n.º 2"
- "art 580 do cpc" → "CPC art. 580"
- "lei 65 de 2003 artigo 3" → "Lei n.º 65/2003 art. 3"
- "decreto lei 15 93 artigo 26" → "DL 15/93 art. 26"
- "código do trabalho" → "CT"
- "regulamento das custas processuais" → "RCP"
- "artigo 50 codigo da estrada" → "CE art. 50"

Input: {raw}
Output (only the citation string, no explanation):"""


def _raw_mentions_article(raw: str) -> bool:
    """Heuristic: does the raw user input mention an article/paragraph/alínea?"""
    return bool(
        re.search(
            r"\b(?:artigo|artigos|arts?\.|art\b|n\.?\s*º\s*s?|alínea|al\.)",
            raw,
            flags=re.IGNORECASE,
        )
    )


def _strip_invented_article(raw: str, llm_output: str) -> str:
    """Safety-net: if the user never mentioned an article, drop any 'art. X' the LLM hallucinated."""
    if not _raw_mentions_article(raw):
        # Strip from first "art." or "artigo" onwards
        m = re.search(r"\s+art\b", llm_output, flags=re.IGNORECASE)
        if m:
            return llm_output[:m.start()].strip()
    return llm_output


async def _llm_canonicalize_legislation_query(raw: str) -> Optional[str]:
    """Ask an LLM to turn a messy Portuguese legislation reference into a canonical string.

    Returns the canonical citation (e.g. "CT art. 2") or None on failure.
    """
    if not http_client or not OPENROUTER_API_KEY:
        return None
    prompt = _LEGISLATION_NORMALIZE_PROMPT.format(raw=raw)
    try:
        resp = await http_client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 60,
            },
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/pt-caselaw-dgsi",
                "X-Title": "pt-caselaw-dgsi",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if the model wrapped the output
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].lstrip()
        content = _strip_invented_article(raw, content)
        # Post-LLM sanity check: does the raw input plausibly refer to the law the LLM chose?
        law, _ = _parse_legislation_ref(content)
        if law and not _validate_llm_law_match(raw, law):
            return None
        return content or None
    except Exception:
        return None


_RERANK_LEGISLATION_PROMPT = """You are a Portuguese legal citation matcher.
Given a user's search query and a list of retrieved legislation references, select which references are ACTUALLY relevant to what the user is looking for.

User query: "{query}"

Retrieved legislation references (ranked by similarity):
{candidates}

Each reference shows: index, citation text, embedding similarity score, and how many court decisions cite it (doc_count).
Prefer references with higher doc_count when multiple variants match the same law/article — a higher doc_count indicates the canonical, well-formed reference.
Ignore references with very low doc_count (e.g. 1–5) if a similar reference exists with a much higher doc_count.

For each reference that genuinely matches the user's intent, output its index (0-based) and a brief reason.
Return ONLY a JSON array in this exact format (no markdown fences, no explanation):
[
  {{"idx": 0, "reason": "Exact match for the requested article, 3854 documents"}},
  {{"idx": 2, "reason": "Same law, related article"}}
]

If none are relevant, return an empty array: []
Do NOT include references that are similar but unrelated.
Do NOT select a low-doc_count variant when a high-doc_count variant of the same article exists.
"""


async def _llm_rerank_legislation(query: str, candidates: list[dict]) -> list[dict]:
    """Ask an LLM to select which retrieved legislation references match the user query.

    Returns a list of {{"idx": int, "reason": str}} objects.
    """
    if not http_client or not OPENROUTER_API_KEY or not candidates:
        return []
    formatted = "\n".join(
        f"  {i}. {c['citation_text']} (similarity: {c['sim']:.3f}, doc_count: {c.get('doc_count', 0)})"
        for i, c in enumerate(candidates)
    )
    prompt = _RERANK_LEGISLATION_PROMPT.format(query=query, candidates=formatted)
    try:
        resp = await http_client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 300,
            },
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/pt-caselaw-dgsi",
                "X-Title": "pt-caselaw-dgsi",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].lstrip()
        result = _json.loads(content)
        if not isinstance(result, list):
            return []
        valid = [
            r for r in result
            if isinstance(r, dict)
            and isinstance(r.get("idx"), int)
            and 0 <= r["idx"] < len(candidates)
        ]
        return valid
    except Exception:
        return []


async def _resolve_legislation_intent(query: str) -> list[dict[str, str]]:
    """Resolve a natural-language legal issue to likely Portuguese provisions."""
    if not http_client or not OPENROUTER_API_KEY:
        return []
    schema = {
        "name": "portuguese_legislation_references",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "citations": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "law": {"type": "string"},
                            "article": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["law", "article", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["citations"],
            "additionalProperties": False,
        },
    }
    prompt = (
        "Identifique as disposições de legislação portuguesa diretamente aplicáveis "
        "à questão. Use a lei ou código canónico e o número exato do artigo. "
        "Devolva no máximo três artigos, apenas quando tiver confiança elevada. "
        "Não devolva doutrina nem jurisprudência.\n\nQuestão: " + query
    )
    try:
        response = await http_client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/pt-caselaw-dgsi",
                "X-Title": "pt-caselaw-dgsi",
            },
            json={
                "model": LEGISLATION_RESOLVER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_schema", "json_schema": schema},
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = _json.loads(content)
        resolved = []
        for citation in parsed.get("citations") or []:
            law = _canonicalize_law(citation["law"])
            article = _normalize_article_prefix(citation["article"])
            if law and article:
                resolved.append({
                    "law": law,
                    "article": article,
                    "reason": citation["reason"],
                })
        return resolved
    except Exception as exc:
        logger.warning("legislation intent resolution failed: %s", exc)
        return []


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
    # One judge has at most a few thousand decisions: rank them exactly. The
    # HNSW walk is capped (HNSW_MAX_SCAN_TUPLES) and would return few of them.
    # "+ 0" keeps the planner from ordering by the HNSW index.
    exact = " + 0" if filters is not None and filters.judge else ""
    sql = (
        f"SELECT doc_id, (1 - ({field} <=> $1::halfvec))::real AS sim "
        f"FROM documents "
        f"WHERE {where} "
        f"ORDER BY ({field} <=> $1::halfvec){exact} "
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


async def _search_vectors_by_text(field_texts: dict[str, str], limit: int,
                                  filters: Optional[Filters]
                                  ) -> dict[str, list[tuple[str, float]]]:
    """Run each vector column against its own query text, embedding every
    distinct text once. Returns {field: [(doc_id, similarity), ...]}."""
    texts = list(dict.fromkeys(field_texts.values()))
    per_text = await asyncio.gather(*(
        _search_vectors(text, [f for f, t in field_texts.items() if t == text],
                        limit, filters)
        for text in texts
    ))
    merged: dict[str, list[tuple[str, float]]] = {}
    for hits in per_text:
        merged.update(hits)
    return {f: merged[f] for f in field_texts if f in merged}


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
# Specific fact-pattern queries often miss relevant decisions that describe the
# same situation in other words. An LLM rewrites the query in the style of the
# stored `semantic_search_query` sentences (what each decision's `embedding`
# is built from) and proposes alternative legal keywords; each variant is
# searched as an extra RRF source. The reranker still sees only the original
# query. Failures or timeouts silently skip expansion.
QUERY_EXPANSION = os.getenv("QUERY_EXPANSION", "false").lower() == "true"
QUERY_EXPANSION_MODEL = "google/gemini-3.1-flash-lite"
QUERY_EXPANSION_TIMEOUT = float(os.getenv("QUERY_EXPANSION_TIMEOUT", "6"))
# RRF weight of each variant source relative to the original query's source.
QUERY_EXPANSION_WEIGHT = float(os.getenv("QUERY_EXPANSION_WEIGHT", "0.5"))

_QUERY_EXPANSION_PROMPT = """És um jurista português especialista em pesquisa de jurisprudência.
Um utilizador procura acórdãos portugueses.

Consulta: {semantic}
Termos-chave do utilizador: {keywords}

Gera variantes para melhorar a recuperação:
1. "reformulacoes": 3 frases em português (cada uma com no máximo 30 palavras) que
   descrevam "que tipo de caso é este", com os factos essenciais e a questão jurídica,
   usando a terminologia que um tribunal usaria (nomes de institutos jurídicos,
   figuras processuais). Cada frase deve usar palavras diferentes das outras.
2. "palavras_chave": 2 conjuntos alternativos de 2 a 4 termos jurídicos curtos
   (sinónimos ou designações técnicas) que um acórdão relevante conteria.
Não inventes factos que alterem o sentido da consulta."""

_QUERY_EXPANSION_SCHEMA = {
    "name": "query_expansion",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reformulacoes": {"type": "array", "items": {"type": "string"}},
            "palavras_chave": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "string"}},
            },
        },
        "required": ["reformulacoes", "palavras_chave"],
        "additionalProperties": False,
    },
}


async def _expand_query(
    semantic: Optional[str], keywords: Optional[str]
) -> tuple[list[str], list[str]]:
    """Return (semantic rephrasings, alternative keyword queries); ([], []) on failure."""
    if not QUERY_EXPANSION or not http_client or not OPENROUTER_API_KEY:
        return [], []
    prompt = _QUERY_EXPANSION_PROMPT.format(
        semantic=semantic or "(não indicada)", keywords=keywords or "(não indicados)"
    )
    try:
        resp = await http_client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            json={
                "model": QUERY_EXPANSION_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "response_format": {"type": "json_schema", "json_schema": _QUERY_EXPANSION_SCHEMA},
            },
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=QUERY_EXPANSION_TIMEOUT,
        )
        resp.raise_for_status()
        parsed = _json.loads(resp.json()["choices"][0]["message"]["content"])
    except Exception as exc:
        logger.warning("query expansion failed; searching original query only: %s", exc)
        return [], []
    rephrasings = [s.strip() for s in parsed.get("reformulacoes", []) if isinstance(s, str) and s.strip()][:3]
    keyword_sets = [
        " ".join(t.strip() for t in terms if isinstance(t, str) and t.strip())
        for terms in parsed.get("palavras_chave", [])[:2]
        if isinstance(terms, list)
    ]
    return rephrasings, [k for k in keyword_sets if k]


async def _search_expansions(
    rephrasings: list[str],
    keyword_sets: list[str],
    fields: list[str],
    use_fts: bool,
    limit: int,
    filters: Optional[Filters],
) -> dict[str, list[tuple[str, float]]]:
    """Search every variant; sources are named '<field>~<n>' and 'fts~<n>'."""
    tasks: dict[str, Any] = {}
    if rephrasings and fields:
        embeddings = await embed_texts(rephrasings)
        for n, emb in enumerate(embeddings, start=1):
            emb_lit = "[" + ",".join(f"{x:.7f}" for x in emb) + "]"
            for field in fields:
                tasks[f"{field}~{n}"] = _vector_query(emb_lit, field, limit, filters)
    if use_fts:
        for n, kw in enumerate(keyword_sets, start=1):
            tasks[f"fts~{n}"] = _search_fts(kw, limit, filters)
    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))


FTS_CANDIDATE_CAP = int(os.getenv("FTS_CANDIDATE_CAP", "1500"))


def _normalize_portuguese_fts_query(query: str) -> str:
    """Match the unaccented Portuguese tsvector while preserving operators."""
    normalized = unicodedata.normalize("NFKD", query)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return normalized.replace("“", '"').replace("”", '"').replace("’", "'")


async def _search_fts(q: str, limit: int,
                      filters: Optional[Filters] = None
                      ) -> list[tuple[str, float]]:
    """Full-text search, ranked over *all* matches.

    `documents.fts` includes the full decision text, so ranking every match
    means reading large TOASTed tsvectors (~20s for a broad query). Matches
    are therefore ranked on `document_fts_core` (keywords, descriptors,
    summary, legal question, ratio — small enough to rank exhaustively).
    Documents matching only in the body text are appended after, from a
    capped candidate set ranked on the full tsvector.
    """
    normalized_q = _normalize_portuguese_fts_query(q)
    # $1 = q, filter params start at $2, then $limit_idx.
    filt_sql, filt_params = _build_filters(filters, start_idx=2, alias="d")
    limit_idx = 2 + len(filt_params)
    if filt_params:
        core_from = "document_fts_core c JOIN documents d ON d.doc_id = c.doc_id"
        core_filter = filt_sql
    else:
        # No user filters: skip the per-match documents lookup.
        core_from = "document_fts_core c"
        core_filter = f"c.doc_id {NOT_DUPLICATE_DOC_IDS}"
    core_sql = (
        "SELECT c.doc_id, ts_rank_cd(c.fts_core, websearch_to_tsquery('portuguese', $1))::real AS rank "
        f"  FROM {core_from} "
        " WHERE c.fts_core @@ websearch_to_tsquery('portuguese', $1) "
        f"   AND {core_filter} "
        " ORDER BY rank DESC, c.doc_id "
        f" LIMIT ${limit_idx}"
    )
    async with db_pool.acquire() as conn:
        core_rows = await conn.fetch(core_sql, normalized_q, *filt_params, limit)
        hits = [(r["doc_id"], float(r["rank"])) for r in core_rows]
        if len(hits) >= limit:
            return hits

        # Body-only matches: $limit_idx = remaining slots, then the cap and
        # the already-returned ids.
        cap_idx = limit_idx + 1
        exclude_idx = limit_idx + 2
        body_sql = (
            "WITH cands AS ("
            "  SELECT d.doc_id, d.fts "
            "    FROM documents d "
            "   WHERE d.fts @@ websearch_to_tsquery('portuguese', $1) "
            f"    AND {filt_sql} "
            f"    AND NOT (d.doc_id = ANY(${exclude_idx}::text[])) "
            f"   LIMIT ${cap_idx}"
            ") "
            "SELECT doc_id, "
            "       ts_rank_cd(fts, websearch_to_tsquery('portuguese', $1))::real AS rank "
            "  FROM cands "
            " ORDER BY rank DESC "
            f" LIMIT ${limit_idx}"
        )
        body_rows = await conn.fetch(
            body_sql, normalized_q, *filt_params, limit - len(hits),
            FTS_CANDIDATE_CAP, [doc_id for doc_id, _ in hits],
        )
    return hits + [(r["doc_id"], float(r["rank"])) for r in body_rows]


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


def _plain_keywords(keyword_query: str) -> str:
    """Drop websearch syntax (quotes, OR) so the reranker reads plain terms."""
    text = re.sub(r"\s+OR\s+", ", ", keyword_query).replace('"', "")
    return re.sub(r"\s+", " ", text).strip()


def _rerank_document_text(doc: dict[str, Any]) -> str:
    """Build a compact Portuguese caselaw representation for reranking."""
    metadata = doc.get("metadata") or {}
    descriptors = metadata.get("legal_descriptors") or []
    citations = metadata.get("legislation_cited") or []
    citation_text = ", ".join(
        " ".join(
            str(value)
            for value in (
                citation.get("law"),
                citation.get("article"),
                citation.get("citation_context"),
            )
            if value
        )
        for citation in citations
        if isinstance(citation, dict)
    )
    fields = [
        f"Tribunal: {doc.get('court_short') or ''}",
        f"Data: {doc.get('decision_date') or ''}",
        f"Domínio jurídico: {doc.get('legal_domain') or ''}",
        f"Resumo: {doc.get('summary') or ''}",
        f"Questão jurídica: {metadata.get('legal_question') or ''}",
        f"Decisão: {metadata.get('decision_outcome') or ''}",
        f"Fundamentação determinante: {metadata.get('ratio_decidendi') or ''}",
        f"Descritores: {', '.join(str(value) for value in descriptors)}",
        f"Legislação citada: {citation_text}",
    ]
    return "\n".join(fields)[:12000]


_rerank_cooldown_until: dict[str, float] = {}


def _retry_after_seconds(response: httpx.Response) -> Optional[float]:
    try:
        value = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return value if value >= 0 else None


async def _post_rerank(
    query: str, documents: list[str]
) -> Optional[tuple[str, list[dict[str, Any]]]]:
    """Rerank with the first model in ``RERANK_MODELS`` that answers.

    Retries 429/5xx and network errors with exponential backoff, then moves
    to the next model. Returns ``(model, results)`` or ``None`` when every
    model failed or is cooling down after a 429.
    """
    for model in RERANK_MODELS:
        if _rerank_cooldown_until.get(model, 0.0) > time.monotonic():
            logger.info("rerank: %s cooling down after 429; skipping", model)
            continue
        response: Optional[httpx.Response] = None
        for attempt in range(RERANK_MAX_ATTEMPTS):
            last_attempt = attempt == RERANK_MAX_ATTEMPTS - 1
            delay = min(
                RERANK_BACKOFF_BASE_SECONDS * (2**attempt),
                RERANK_MAX_RETRY_DELAY_SECONDS,
            )
            try:
                response = await http_client.post(
                    f"{OPENROUTER_BASE}/rerank",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/pt-caselaw-dgsi",
                        "X-Title": "pt-caselaw-dgsi",
                    },
                    json={
                        "model": model,
                        "query": query,
                        "documents": documents,
                        "top_n": len(documents),
                    },
                    timeout=RERANK_TIMEOUT,
                )
            except httpx.RequestError as exc:
                response = None
                if last_attempt:
                    logger.warning("rerank: %s network error: %s", model, exc)
                    break
                await asyncio.sleep(delay)
                continue
            if response.status_code not in RERANK_RETRY_STATUSES or last_attempt:
                break
            retry_after = _retry_after_seconds(response)
            if retry_after is not None:
                delay = min(retry_after, RERANK_MAX_RETRY_DELAY_SECONDS)
            await asyncio.sleep(delay)

        if response is None:
            continue
        if response.status_code == 200:
            try:
                results = response.json().get("results")
            except ValueError:
                results = None
            if isinstance(results, list):
                if model != RERANK_MODELS[0]:
                    logger.info("rerank: served by fallback model %s", model)
                return model, results
            logger.warning("rerank: %s returned no results list", model)
            continue
        if response.status_code == 429:
            cooldown = min(
                max(_retry_after_seconds(response) or 0.0, RERANK_COOLDOWN_SECONDS),
                RERANK_MAX_COOLDOWN_SECONDS,
            )
            _rerank_cooldown_until[model] = time.monotonic() + cooldown
            logger.warning(
                "rerank: %s rate limited (429); skipping it for %.0fs", model, cooldown
            )
        else:
            logger.warning(
                "rerank: %s HTTP %d: %s",
                model, response.status_code, response.text[:200],
            )
    return None


async def _rerank_candidates(
    semantic_query: Optional[str],
    keyword_query: Optional[str],
    merged: list[tuple[str, float]],
    docs: dict[str, dict],
    rrf_k: int,
    rerank_weight: Optional[float] = None,
    lexical_promotion_ids: Optional[set[str]] = None,
    question_only: bool = False,
) -> list[tuple[str, float]]:
    """Add a dedicated reranker rank to the existing RRF score.

    With ``question_only`` the reranker reads only the semantic query and its
    order replaces the fused order.

    Failures leave the original order untouched. The API contract remains the
    same: ``hybrid_score`` is still a fused rank score and source diagnostics
    continue to describe the four retrieval sources.
    """
    effective_weight = RERANK_WEIGHT if rerank_weight is None else rerank_weight
    question_only = question_only and bool(semantic_query)
    if (
        effective_weight <= 0
        or not RERANK_MODELS
        or not OPENROUTER_API_KEY
        or not http_client
        or len(merged) < 2
    ):
        return merged

    candidate_ids = [doc_id for doc_id, _ in merged if doc_id in docs]
    if len(candidate_ids) < 2:
        return merged
    documents = [_rerank_document_text(docs[doc_id]) for doc_id in candidate_ids]
    query_parts = [] if question_only else [
        "Ordene acórdãos portugueses pela relevância para a consulta. "
        "Dê prioridade aos acórdãos que decidam diretamente a questão e aos "
        "factos específicos pedidos, não apenas aos que partilhem vocabulário.",
    ]
    if semantic_query:
        query_parts.append(f"Consulta: {semantic_query}")
    if keyword_query and not question_only:
        query_parts.append(f"Termos jurídicos essenciais: {_plain_keywords(keyword_query)}")

    try:
        reranked_by = await _post_rerank("\n".join(query_parts), documents)
        if reranked_by is None:
            return merged
        model, results = reranked_by
        rerank_positions = {
            candidate_ids[item["index"]]: position
            for position, item in enumerate(results, start=1)
            if isinstance(item.get("index"), int)
            and 0 <= item["index"] < len(candidate_ids)
        }
        if len(rerank_positions) != len(candidate_ids):
            logger.warning(
                "rerank (%s) returned %d/%d candidates; preserving original order",
                model, len(rerank_positions), len(candidate_ids),
            )
            return merged
        rescored = [
            (doc_id, (0.0 if question_only else score)
             + effective_weight / (rrf_k + rerank_positions[doc_id]))
            for doc_id, score in merged
            if doc_id in rerank_positions
        ]
        if RERANK_LEXICAL_GUARD and lexical_promotion_ids is not None and len(merged) > 5:
            original_top_ids = {doc_id for doc_id, _ in merged[:5]}
            top_floor = min(
                score for doc_id, score in rescored if doc_id in original_top_ids
            )
            rescored = [
                (
                    doc_id,
                    min(score, top_floor - 1e-9)
                    if doc_id not in original_top_ids
                    and doc_id not in lexical_promotion_ids
                    else score,
                )
                for doc_id, score in rescored
            ]
        return sorted(rescored, key=lambda item: (-item[1], item[0]))
    except Exception as exc:
        logger.warning("rerank failed; preserving retrieval order: %s", exc)
        return merged


async def _rerank_legislation_rows(
    query: str,
    rows: list[asyncpg.Record],
) -> list[dict[str, Any]]:
    """Rerank date-ordered legislation matches by substantive relevance."""
    docs = [_row_to_dict(row) for row in rows]
    docs_by_id = {doc["doc_id"]: doc for doc in docs}
    date_ranking = [
        (doc["doc_id"], 1 / (50 + rank))
        for rank, doc in enumerate(docs, start=1)
    ]
    reranked = await _rerank_candidates(
        query,
        None,
        date_ranking,
        docs_by_id,
        50,
        rerank_weight=LEGISLATION_RERANK_WEIGHT,
    )
    return [docs_by_id[doc_id] for doc_id, _ in reranked]


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
        "Returns `healthy` when the HTTP client is initialised and the "
        "database answers a trivial query within 2 seconds; otherwise "
        "returns **503** with `status: unhealthy` and the failing component."
    ),
    responses={503: {"description": "Database unreachable or not initialised."}},
)
async def health():
    if db_pool is None or http_client is None:
        return JSONResponse({"status": "unhealthy", "reason": "not initialised"}, status_code=503)
    try:
        async with db_pool.acquire(timeout=2) as conn:
            await conn.fetchval("SELECT 1", timeout=2)
    except Exception as exc:
        logger.warning("health check: database unreachable: %s", exc)
        return JSONResponse({"status": "unhealthy", "reason": "database"}, status_code=503)
    return {"status": "healthy"}


STATS_CACHE_TTL = int(os.getenv("STATS_CACHE_TTL", "600"))  # seconds
_stats_cache: Optional[tuple[float, dict]] = None


@app.get(
    "/stats",
    tags=["info"],
    summary="Corpus and configuration statistics",
    description=(
        "Returns the total number of documents, how many have each of the "
        "three embedding columns populated, the embedding model in use, and "
        "the HNSW `ef_search` setting. Use this to sanity-check ingestion "
        "coverage before relying on a search variant. Cached in-memory for "
        "`STATS_CACHE_TTL` seconds (default 600)."
    ),
)
async def stats():
    # The counts scan the whole documents table; they only change when the
    # ETL runs, so serve them from a cache.
    global _stats_cache
    now = time.time()
    if _stats_cache and (now - _stats_cache[0]) < STATS_CACHE_TTL:
        return {**_stats_cache[1], "cached": True,
                "cache_age_seconds": int(now - _stats_cache[0])}
    async with db_pool.acquire() as conn:
        counts = await conn.fetchrow(
            "SELECT count(*) AS total, "
            "       count(*) FILTER (WHERE duplicate_of IS NOT NULL) AS duplicates, "
            "       count(*) FILTER (WHERE embedding IS NOT NULL) AS embedding, "
            "       count(*) FILTER (WHERE embedding_context IS NOT NULL) AS embedding_context, "
            "       count(*) FILTER (WHERE embedding_ratio IS NOT NULL) AS embedding_ratio "
            "  FROM documents"
        )
    payload = {
        "total_documents": counts["total"],
        "duplicates_excluded_from_search": counts["duplicates"],
        "embeddings": {
            "embedding": counts["embedding"],
            "embedding_context": counts["embedding_context"],
            "embedding_ratio": counts["embedding_ratio"],
        },
        "embedding_model": EMBEDDING_MODEL,
        "hnsw_ef_search": HNSW_EF_SEARCH,
    }
    _stats_cache = (now, payload)
    return {**payload, "cached": False, "cache_age_seconds": 0}


def _resolve_queries(req: SearchRequest, need_sem: bool, need_fts: bool
                     ) -> tuple[Optional[str], Optional[str]]:
    """Extract the semantic and keyword query strings from the request.
    Raises 400 if a required slot is missing."""
    sem = req.q_semantic
    kw = " ".join(req.q_keywords) if req.q_keywords is not None else None
    if need_sem and not sem:
        raise HTTPException(400, "Provide `q_semantic` for vector search")
    if need_fts and not kw:
        raise HTTPException(400, "Provide `q_keywords` for keyword search")
    return sem, kw


def _enabled_vector_fields(weights: SearchWeights) -> list[str]:
    return [f for f in EMBEDDING_FIELDS if getattr(weights, f) > 0]


def _vector_field_texts(req: SearchRequest, vec_fields: list[str]) -> dict[str, str]:
    """Text to embed per enabled vector column: its own text from
    `q_semantic_fields`, else `q_semantic`. Columns with neither are omitted."""
    own = req.q_semantic_fields
    out: dict[str, str] = {}
    for field in vec_fields:
        text = ((getattr(own, field) or "") if own else "").strip() or (req.q_semantic or "")
        if text:
            out[field] = text
    return out


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
        "lookups. Per-source ranks are combined with weighted "
        "**Reciprocal Rank Fusion**, then the leading candidates are reranked "
        "by a Portuguese-capable legal-text reranker:\n\n"
        "```\nhybrid_score(d) = Σ_source  weight[source] / (rrf_k + rank_source(d))\n```\n\n"
        "**Best practices:**\n\n"
        "- Supply **both** `q_semantic` and `q_keywords` for maximum recall. "
        "Under the default weights both are required — a missing one returns "
        "400; disable a source explicitly via `weights` instead.\n"
        "- Write `q_semantic` as a full legal sentence or question: "
        "`'despedimento sem justa causa por uso indevido de email corporativo'` "
        "outperforms `'despedimento email'`.\n"
        "- Extract 3–6 discriminating nouns/verbs for `q_keywords`. They feed a "
        "GIN FTS index with Portuguese stemming + unaccent and do not need to "
        "mirror `q_semantic` word-for-word.\n"
        "- Apply court and date filters to narrow results before widening `limit`.\n"
        "- Set `is_auj=true` to retrieve only AUJs (binding uniformisation precedents).\n"
        "- To bias ranking: boost `embedding_ratio` (legal reasoning column) for "
        "doctrinal questions; boost `embedding_context` (facts + outcome) for "
        "factual similarity searches.\n"
        "- Call `GET /filters` first to discover valid `court_short` codes and the "
        "corpus date range.\n\n"
        "Use the `weights` object to disable a source (set to 0) or to bias "
        "the ranking toward a particular signal.\n\n"
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
    field_texts = _vector_field_texts(req, vec_fields)
    sem_q, kw_q = _resolve_queries(
        req, need_sem=len(field_texts) < len(vec_fields), need_fts=use_fts
    )
    # Reranking and expansion read the caller's question; without q_semantic
    # fall back to the first per-column text.
    rank_q = sem_q or next(iter(field_texts.values()), None)
    over = req.limit * req.overfetch
    effective_filters = req.resolved_filters()

    try:
        # Query expansion runs concurrently with the original-query searches.
        expansion_task = asyncio.create_task(_expand_query(rank_q, kw_q))
        try:
            if vec_fields and use_fts:
                per_vec, fts_hits = await asyncio.gather(
                    _search_vectors_by_text(field_texts, over, effective_filters),
                    _search_fts(kw_q, over, effective_filters),
                )
            elif vec_fields:
                per_vec = await _search_vectors_by_text(field_texts, over, effective_filters)
                fts_hits = []
            else:
                per_vec = {}
                fts_hits = await _search_fts(kw_q, over, effective_filters)
        except BaseException:
            expansion_task.cancel()
            raise

        per_source: dict[str, list[tuple[str, float]]] = dict(per_vec)
        if use_fts:
            per_source["fts"] = fts_hits
        weights = {f: getattr(req.weights, f) for f in vec_fields}
        if use_fts:
            weights["fts"] = req.weights.fts

        rephrasings, keyword_sets = await expansion_task
        if rephrasings or keyword_sets:
            expanded = await _search_expansions(
                rephrasings, keyword_sets, vec_fields, use_fts, over, effective_filters
            )
            per_source.update(expanded)
            for source in expanded:
                base = source.split("~", 1)[0]
                weights[source] = weights[base] * QUERY_EXPANSION_WEIGHT

        merged, ranks = _rrf_merge_multi(per_source, weights, k=req.rrf_k)
        # Expose ranks for the four documented sources only.
        ranks = {
            doc_id: {s: r for s, r in doc_ranks.items() if s in ALL_SOURCES} or None
            for doc_id, doc_ranks in ranks.items()
        }
        rerank_count = min(len(merged), max(req.limit, RERANK_CANDIDATES))
        docs = await _fetch_docs([d for d, _ in merged[:rerank_count]])
        merged = await _rerank_candidates(
            rank_q,
            kw_q,
            merged[:rerank_count],
            docs,
            req.rrf_k,
            lexical_promotion_ids={doc_id for doc_id, _ in fts_hits},
            question_only=req.rerank_mode == "question",
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            504,
            "Database query timed out. Try adding courts/date filters, lowering limit, "
            "or disabling one search source via weights.",
        ) from exc
    results = _build_results(merged, docs, per_source, ranks,
                             req.limit, include_hybrid=True)
    sources_used = [*vec_fields] + (["fts"] if use_fts else [])
    return SearchResponse(query_semantic=sem_q, query_keywords=kw_q,
                          count=len(results), sources_used=sources_used,
                          mode="hybrid", filters=effective_filters, results=results)



FILTERS_CACHE_TTL = int(os.getenv("FILTERS_CACHE_TTL", "3600"))  # seconds
_filters_cache: Optional[tuple[float, dict]] = None


async def _compute_filters_payload() -> dict:
    async with db_pool.acquire() as conn:
        court_rows = await conn.fetch(
            "SELECT court_short AS value, count(*) AS count FROM documents "
            " WHERE court_short IS NOT NULL "
            " GROUP BY court_short ORDER BY count DESC"
        )
        date_row = await conn.fetchrow(
            "SELECT min(decision_date) AS min, max(decision_date) AS max "
            "  FROM documents"
        )
        is_auj_rows = await conn.fetch(
            "SELECT is_auj AS value, count(*) AS count FROM documents "
            " WHERE is_auj IS NOT NULL "
            " GROUP BY is_auj ORDER BY value DESC"
        )
        area_rows = await conn.fetch(
            "SELECT area AS value, count(*) AS count "
            "  FROM documents d "
            "  JOIN legal_domain_map m ON m.raw = d.legal_domain, "
            "       unnest(m.areas) AS area "
            " WHERE d.duplicate_of IS NULL "
            " GROUP BY area ORDER BY count DESC"
        )
    counts_by_area = {r["value"]: r["count"] for r in area_rows}
    return {
        "courts": [{"value": r["value"], "count": r["count"]} for r in court_rows],
        "decision_date": {
            "min": date_row["min"].isoformat() if date_row["min"] else None,
            "max": date_row["max"].isoformat() if date_row["max"] else None,
        },
        "is_auj": [{"value": r["value"], "count": r["count"]} for r in is_auj_rows],
        "legal_areas": [
            {"value": area, "count": counts_by_area.get(area, 0)} for area in CANONICAL_AREAS
        ],
    }


@app.get(
    "/filters",
    tags=["info"],
    summary="Discover available courts and date range for search filters",
    description=(
        "Returns the values you can pass to the `courts`, `from_date`/`to_date`, "
        "and `is_auj` filter fields on `POST /search`:\n\n"
        "* `courts` — every `court_short` code present in the corpus with "
        "document counts. Pass these directly as `courts: [\"STJ\", \"TRP\"]`.\n"
        "* `decision_date` — `min` and `max` dates in the corpus, useful for "
        "bounding `from_date` / `to_date`.\n"
        "* `is_auj` — count of binding-precedent (AUJ) decisions vs regular ones.\n"
        "* `legal_areas` — canonical legal areas with document counts. Pass one "
        "as `filters.legal_domain`.\n\n"
        "**Caching** — results are cached in-memory for `FILTERS_CACHE_TTL` "
        "seconds (default 3600). The response echoes `cached` and "
        "`cache_age_seconds`. Pass `refresh=true` to force a recompute."
    ),
)
async def get_filters(
    refresh: bool = Query(
        False,
        description="Bypass the in-memory cache and recompute from the database.",
    ),
):
    global _filters_cache
    now = time.time()
    if not refresh and _filters_cache and (now - _filters_cache[0]) < FILTERS_CACHE_TTL:
        return {**_filters_cache[1], "cached": True,
                "cache_age_seconds": int(now - _filters_cache[0])}
    payload = await _compute_filters_payload()
    _filters_cache = (now, payload)
    return {**payload, "cached": False, "cache_age_seconds": 0}


class JudgeCourt(BaseModel):
    court_short: Optional[str] = Field(description="Court code, as in the `court` filter.")
    count: int = Field(description="Decisions of this judge as rapporteur in that court.")


class JudgeSuggestion(BaseModel):
    """One judge rapporteur, with every spelling that shares its `key`."""
    name: str = Field(description="Most frequent spelling. Pass it in `filters.judge`.")
    key: str = Field(description="Accent-, case- and punctuation-insensitive form used for matching.")
    count: int = Field(description="Decisions with this judge as rapporteur (duplicates excluded).")
    courts: list[JudgeCourt] = Field(description="Courts where the judge was rapporteur, most decisions first.")
    first_date: Optional[date] = None
    last_date: Optional[date] = None


class JudgesResponse(BaseModel):
    count: int
    results: list[JudgeSuggestion]


@app.get(
    "/judges",
    response_model=JudgesResponse,
    tags=["info"],
    summary="Autocomplete judge rapporteurs (relatores)",
    description=(
        "Judges whose name contains every word of `q` as the start of a word, "
        "ignoring case and accents (`calejo`, `j correia`). Spellings that "
        "differ only in case or accents are grouped. Abbreviated forms "
        "(`E. Sequeira` / `Eugénio Sequeira`) stay separate entries; pass "
        "several names in `filters.judge` to search them together. Sorted by "
        "number of decisions."
    ),
)
async def get_judges(
    q: str = Query(..., min_length=2, max_length=100, description="Part of the judge's name."),
    court: Optional[list[str]] = Query(None, description="Only count decisions of these courts."),
    limit: int = Query(10, ge=1, le=50),
):
    tokens = query_tokens(q)
    if not tokens:
        return JudgesResponse(count=0, results=[])
    # Tokens are A–Z/0–9 words (judge_key), so LIKE has nothing to escape.
    conditions = [f"(' ' || judge_key) LIKE ${i + 1}" for i in range(len(tokens))]
    params: list[Any] = [f"% {token}%" for token in tokens]
    if court:
        conditions.append(f"court_short = ANY(${len(params) + 1}::text[])")
        params.append(court)
    sql = (
        "SELECT judge_key, court_short, judge_name, doc_count, first_date, last_date "
        "  FROM judge_directory "
        f" WHERE {' AND '.join(conditions)}"
    )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    grouped: dict[str, list[asyncpg.Record]] = {}
    for row in rows:
        grouped.setdefault(row["judge_key"], []).append(row)
    results = []
    for key, court_rows in grouped.items():
        court_rows.sort(key=lambda r: r["doc_count"], reverse=True)
        firsts = [r["first_date"] for r in court_rows if r["first_date"]]
        lasts = [r["last_date"] for r in court_rows if r["last_date"]]
        results.append(JudgeSuggestion(
            name=court_rows[0]["judge_name"],
            key=key,
            count=sum(r["doc_count"] for r in court_rows),
            courts=[JudgeCourt(court_short=r["court_short"], count=r["doc_count"]) for r in court_rows],
            first_date=min(firsts) if firsts else None,
            last_date=max(lasts) if lasts else None,
        ))
    results.sort(key=lambda r: (-r.count, r.key))
    return JudgesResponse(count=min(len(results), limit), results=results[:limit])


class BrowseRequest(BaseModel):
    """Request body for `POST /browse`."""
    filters: Filters = Field(description="Filters to list; `judge` is required.")
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0, le=10000)


@app.post(
    "/browse",
    response_model=SearchResponse,
    tags=["search"],
    summary="List a judge's decisions by date, without a query",
    description=(
        "Returns the decisions matching `filters`, newest first. Requires "
        "`filters.judge` so the listing stays small; combine it with court, "
        "dates, AUJ and the other filters. Use `offset` to page."
    ),
)
async def browse(req: BrowseRequest):
    if not req.filters.judge:
        raise HTTPException(400, "Provide `filters.judge` to browse without a query")
    filt_sql, filt_params = _build_filters(req.filters, start_idx=1)
    limit_idx = len(filt_params) + 1
    sql = (
        "SELECT doc_id FROM documents "
        f"WHERE {filt_sql} "
        "ORDER BY decision_date DESC NULLS LAST, doc_id "
        f"LIMIT ${limit_idx} OFFSET ${limit_idx + 1}"
    )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, *filt_params, req.limit, req.offset)
    ids = [r["doc_id"] for r in rows]
    docs = await _fetch_docs(ids)
    results = [SearchResult(**docs[doc_id]) for doc_id in ids if doc_id in docs]
    return SearchResponse(count=len(results), sources_used=[], mode="browse",
                          filters=req.filters, results=results)


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
        "Set `include_full_text=true` to also receive the `full_text` field "
        "with the complete integral text of the decision as scraped from DGSI. "
        "Omitted by default because it can be several hundred kilobytes.\n\n"
        "Typical usage: after a `/search` call, pick the `doc_id` of an "
        "interesting result and hydrate it here for full context."
    ),
    responses={
        404: {"description": "No document found for the given `doc_id`."},
    },
)
async def get_document(
    doc_id: str = FastAPIPath(
        ...,
        description="Stable opaque document identifier returned by any /search endpoint.",
        examples=["3a8c0d2e9f1b4a7e8d6c5b4a3f2e1d0c"],
    ),
    include_full_text: bool = Query(
        False,
        description=(
            "When `true`, the response includes a `full_text` field with the "
            "complete integral text of the decision as scraped from DGSI. "
            "Omitted by default to keep responses compact."
        ),
    ),
):
    docs = await _fetch_docs([doc_id])
    if doc_id not in docs:
        raise HTTPException(404, "Document not found")
    doc = docs[doc_id]
    if include_full_text:
        async with db_pool.acquire() as conn:
            full_text = await conn.fetchval(
                "SELECT full_text FROM documents WHERE doc_id = $1", doc_id
            )
        doc["full_text"] = full_text
    return doc


# ---------------------------------------------------------------------------
# Cited decision resolution
# ---------------------------------------------------------------------------

from api.citations import PROCESS_KEY_SQL, court_code, pick_candidate, process_key  # noqa: E402


class CitedDecision(BaseModel):
    """One entry of `metadata.jurisprudence_cited`, as the extractor writes it."""
    process_number: Optional[str] = Field(None, max_length=200)
    court_abbreviation: Optional[str] = Field(None, max_length=40)
    court_name: Optional[str] = Field(None, max_length=200)
    decision_date: Optional[str] = Field(None, max_length=20)


class ResolveCitationsRequest(BaseModel):
    citations: list[CitedDecision] = Field(..., max_length=50)


class ResolvedCitation(BaseModel):
    index: int = Field(description="Position of the citation in the request.")
    doc_id: Optional[str] = Field(
        None, description="Matching corpus document, or null when not in the corpus."
    )
    court_short: Optional[str] = None
    process_number: Optional[str] = None
    decision_date: Optional[date] = None
    url: Optional[str] = None


class ResolveCitationsResponse(BaseModel):
    results: list[ResolvedCitation]


@app.post(
    "/documents/resolve-citations",
    response_model=ResolveCitationsResponse,
    tags=["documents"],
    summary="Map cited decisions to corpus documents",
    description=(
        "Takes up to 50 entries of a document's `metadata.jurisprudence_cited` "
        "and returns, for each one in order, the `doc_id` of the cited decision "
        "when it is in the corpus (null otherwise). Matching is on the "
        "process number with whitespace removed and case ignored; court and "
        "date pick between documents sharing a number, and an ambiguous "
        "match returns null rather than a guess. Duplicates are skipped."
    ),
)
async def resolve_citations(req: ResolveCitationsRequest):
    keys = [process_key(c.process_number) for c in req.citations]
    wanted = sorted({k for k in keys if k})
    by_key: dict[str, list[dict]] = {}
    if wanted:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT doc_id, url, court_short, process_number, decision_date, "
                f"{PROCESS_KEY_SQL} AS key FROM documents "
                f"WHERE {PROCESS_KEY_SQL} = ANY($1::text[]) AND duplicate_of IS NULL",
                wanted,
            )
        for r in rows:
            by_key.setdefault(r["key"], []).append(dict(r))

    results: list[ResolvedCitation] = []
    for i, (cite, key) in enumerate(zip(req.citations, keys)):
        match = pick_candidate(
            by_key.get(key, []) if key else [],
            court_code(cite.court_abbreviation, cite.court_name),
            cite.decision_date,
        )
        if match:
            results.append(ResolvedCitation(
                index=i,
                doc_id=match["doc_id"],
                court_short=match["court_short"],
                process_number=match["process_number"],
                decision_date=match["decision_date"],
                url=match["url"],
            ))
        else:
            results.append(ResolvedCitation(index=i))
    return ResolveCitationsResponse(results=results)


# ---------------------------------------------------------------------------
# Legislation article search
# ---------------------------------------------------------------------------

from api._law_utils import _LAW_ABBREV, _canonicalize_law, _clean_article_text  # noqa: E402

# All known law string tokens (abbreviations + canonical names), longest first
# so that e.g. "cpp" matches before "cp" and "cpta" before "cpt".
_LAW_TOKENS: list[str] = sorted(
    list(_LAW_ABBREV.keys()) + [v.lower() for v in set(_LAW_ABBREV.values())],
    key=len,
    reverse=True,
)

# Words that are too generic to distinguish one law from another.
_GENERIC_LAW_WORDS: set[str] = {
    "codigo", "de", "do", "da", "dos", "das", "e", "o", "a", "os", "as",
    "no", "na", "nos", "nas", "pelo", "pela", "pelos", "pelas",
    "n", "n.", "n.", "n.o", "nº", "lei", "decreto", "regulamento", "processo",
    "tribunal", "tribunais", "artigo", "art", "arts", "art.", "alinea", "al",
}


def _strip_accents_local(s: str) -> str:
    """Remove Portuguese diacritics for loose matching."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _validate_llm_law_match(raw: str, law: str) -> bool:
    """Post-LLM sanity check: does the raw input plausibly refer to this law?

    Rejects obvious hallucinations (e.g. 'codigo da estrada' → 'CT').
    """
    if not raw or not law:
        return False

    raw_norm = _strip_accents_local(raw).lower()
    law_norm = _strip_accents_local(law).lower()

    # 1. Check if any abbreviation for this exact law appears in raw
    abbrs = [
        k for k, v in _LAW_ABBREV.items()
        if _strip_accents_local(v).lower() == law_norm
    ]
    for abbr in abbrs:
        if re.search(r"\b" + re.escape(abbr.lower()) + r"\b", raw_norm):
            return True

    # 2. Full canonical name appears verbatim
    if law_norm in raw_norm:
        return True

    # 3. Meaningful keywords from the law name
    law_words = [
        w for w in law_norm.split()
        if w not in _GENERIC_LAW_WORDS and len(w) > 2
    ]
    for w in law_words:
        if w in raw_norm:
            return True

    # 4. Statute number match (e.g. "65/2003")
    num_match = re.search(r"\b(\d+(?:-[A-Za-z])?)/(\d{2,4})\b", law_norm)
    if num_match:
        full_num = f"{num_match.group(1)}/{num_match.group(2)}"
        if full_num in raw_norm:
            return True
        # Accept when both number and year appear separately in raw
        if num_match.group(1) in raw_norm and num_match.group(2) in raw_norm:
            return True

    return False


def _normalize_article_prefix(art_str: str) -> Optional[str]:
    """Normalize a user-supplied article string to the canonical LIKE prefix
    used in the DB (e.g. '394.º', '394.º, n.º 2', '394.º, n.º 2, alínea b)').

    Steps:
      1. Apply the extractor's _clean_article_text (ordinal marks, alínea, n.º).
      2. Ensure the leading article number carries '.º'.
      3. Normalise the 'n.º' separator to ', n.º ' (comma-space).
      4. Normalise the 'alínea' separator similarly.
    """
    if not art_str:
        return None
    s = _clean_article_text(art_str)
    if not s:
        return None
    # Ensure leading article number has .º (e.g. '394' → '394.º', '394.º' unchanged).
    # Check first whether .º is already present; if not, add it after the whole digit run.
    if not re.match(r"^\d+\.º(?:-[A-Za-z])?", s):
        s = re.sub(r"^(\d+(?:-[A-Za-z])?)", r"\1.º", s)
    # Normalise 'N.º M' → ', n.º M' (space-separated after .º)
    s = re.sub(r"(?<=\.º)\s+n\.º\s+", ", n.º ", s)
    # Normalise alínea separator
    s = re.sub(r",?\s+alínea\s+", ", alínea ", s)
    return s.strip() or None


def _parse_legislation_ref(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Parse a free-form article citation into (canonical_law, article_like_prefix).

    Handles mixed Portuguese user input such as:
      'CT art. 394'                → ('Código do Trabalho', '394.º')
      'artigo 394.º CT'            → ('Código do Trabalho', '394.º')
      'CPC art. 640, n.º 3'       → ('Código de Processo Civil', '640.º, n.º 3')
      'artigo 394 n.º 2 al. b) CT' → ('Código do Trabalho', '394.º, n.º 2, alínea b)')
      'Código Civil 483'           → ('Código Civil', '483.º')
      'Lei n.º 65/2003 art. 5'    → ('Lei n.º 65/2003', '5.º')
      'DL 15/93 art. 26'          → ('Decreto-Lei n.º 15/93', '26.º')
      'CT'                         → ('Código do Trabalho', None)  ← any article

    Returns (None, None) when no law token is found.
    Returns (law, None) when a law is found but no article part remains.
    """
    s = raw.strip()
    law_found: Optional[str] = None
    law_span: tuple[int, int] = (0, 0)

    # 1. Statute patterns first (Decreto-Lei / DL, Lei)
    for pat, tpl in [
        (
            r"\b(?:decreto[\s\-]?lei|dl)\s*(?:n[°º.\s]*)?\s*"
            r"(\d+(?:-[A-Za-z])?/\d+)\b",
            "Decreto-Lei n.º {}",
        ),
        (
            r"\blei\s*(?:n[°º.\s]*)?\s*(\d+(?:-[A-Za-z])?/\d+)\b",
            "Lei n.º {}",
        ),
    ]:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            law_found = tpl.format(m.group(1))
            law_span = (m.start(), m.end())
            break

    # 2. Known abbreviations / full canonical names (longest first)
    if not law_found:
        s_lower = s.lower()
        for tok in _LAW_TOKENS:
            idx = s_lower.find(tok)
            if idx == -1:
                continue
            before_ok = idx == 0 or not s_lower[idx - 1].isalpha()
            after_ok = (
                idx + len(tok) >= len(s_lower)
                or not s_lower[idx + len(tok)].isalpha()
            )
            if before_ok and after_ok:
                law_found = _canonicalize_law(s[idx: idx + len(tok)])
                law_span = (idx, idx + len(tok))
                break

    if not law_found:
        return None, None

    # 3. Remove law token; whatever remains is the article reference
    start, end = law_span
    art_raw = (s[:start] + " " + s[end:]).strip()

    # Strip "artigo"/"art." keywords
    art_raw = re.sub(r"\b(?:artigos?|arts?\.?)\s*", "", art_raw, flags=re.IGNORECASE)
    # Strip Portuguese connectors that may dangle after law removal ("do", "da", …)
    art_raw = re.sub(
        r"\b(?:do|da|dos|das|de|no|na|nos|nas|pelo|pela|pelos|pelas)\b",
        " ", art_raw, flags=re.IGNORECASE,
    )
    art_raw = re.sub(r"\s+", " ", art_raw).strip().strip(".,;:")

    if not art_raw:
        return law_found, None

    return law_found, _normalize_article_prefix(art_raw)


def _article_condition(
    law: str, article_prefix: Optional[str], idx: int
) -> tuple[str, list[Any], int]:
    """Build the SQL fragment for one (law, article_prefix) constraint.

    The GIN containment check on the law name uses the existing
    `idx_documents_metadata` (jsonb_path_ops) index to narrow the row set;
    the EXISTS+LIKE on the article is then a cheap sequential scan of that
    smaller set.

    Returns (sql_fragment, params, next_param_idx).
    """
    params: list[Any] = []

    # GIN index-friendly containment check (jsonb_path_ops supports @>).
    gin_sql = (
        "metadata @> jsonb_build_object("
        f"  'legislation_cited', jsonb_build_array(jsonb_build_object('law', ${idx}::text))"
        ")"
    )
    params.append(law)
    idx += 1

    if article_prefix:
        exists_sql = (
            f"EXISTS ("
            f"  SELECT 1 FROM jsonb_array_elements(metadata->'legislation_cited') _el "
            f"  WHERE _el->>'law' = ${idx} AND _el->>'article' LIKE ${idx + 1}"
            f")"
        )
        params.append(law)
        params.append(article_prefix + "%")
        idx += 2
        sql = f"({gin_sql} AND {exists_sql})"
    else:
        sql = gin_sql

    return sql, params, idx


def _build_citation_filters(
    f: Optional[Filters], start_idx: int, alias: str = "c"
) -> tuple[str, list[Any], int]:
    """Build filters that are denormalized onto the citation lookup table."""
    if f is None:
        return "", [], start_idx
    prefix = f"{alias}."
    clauses: list[str] = []
    params: list[Any] = []
    idx = start_idx
    if f.court:
        clauses.append(f"{prefix}court_short = ANY(${idx}::text[])")
        params.append(f.court)
        idx += 1
    if f.legal_domain:
        clause, param = _legal_domain_clause(f"{prefix}legal_domain", f.legal_domain, idx)
        clauses.append(clause)
        params.append(param)
        idx += 1
    if f.is_auj is not None:
        clauses.append(f"{prefix}is_auj = ${idx}")
        params.append(f.is_auj)
        idx += 1
    if f.date_from is not None:
        clauses.append(f"{prefix}decision_date >= ${idx}")
        params.append(f.date_from)
        idx += 1
    if f.date_to is not None:
        clauses.append(f"{prefix}decision_date <= ${idx}")
        params.append(f.date_to)
        idx += 1
    return " AND ".join(clauses), params, idx


def _build_document_json_filters(
    f: Optional[Filters], start_idx: int, alias: str = "d"
) -> tuple[str, list[Any], int]:
    """Build document-only JSON metadata filters not present on citations."""
    if f is None:
        return "", [], start_idx
    prefix = f"{alias}."
    clauses: list[str] = []
    params: list[Any] = []
    idx = start_idx
    if f.decision_type:
        clauses.append(f"{prefix}metadata->>'decision_type' = ANY(${idx}::text[])")
        params.append(f.decision_type)
        idx += 1
    if f.extraction_confidence:
        clauses.append(f"{prefix}metadata->>'extraction_confidence' = ANY(${idx}::text[])")
        params.append(f.extraction_confidence)
        idx += 1
    clause, points_params, idx = _disability_points_clause(f, idx, prefix)
    if clause:
        clauses.append(clause)
        params.extend(points_params)
    clause, judge_params, idx = _judge_clause(f, idx, prefix)
    if clause:
        clauses.append(clause)
        params.extend(judge_params)
    return " AND ".join(clauses), params, idx


def _citation_ref_select(
    law: str,
    article_prefix: Optional[str],
    match_key: int,
    start_idx: int,
    filters: Optional[Filters],
) -> tuple[str, list[Any], int]:
    """Build one indexed lookup-table SELECT for a law/article constraint."""
    clauses = [f"c.law = ${start_idx}"]
    params: list[Any] = [law]
    idx = start_idx + 1
    if article_prefix:
        clauses.append("c.article <> ''")
        clauses.append(f"(c.article = ${idx} OR c.article LIKE ${idx + 1})")
        params.extend([article_prefix, article_prefix + ",%"])
        idx += 2
    filt_sql, filt_params, idx = _build_citation_filters(filters, idx, alias="c")
    if filt_sql:
        clauses.append(filt_sql)
        params.extend(filt_params)
    sql = (
        f"SELECT c.doc_id, c.decision_date, {match_key} AS match_key "
        f"FROM {LEGISLATION_CITATIONS_TABLE} c "
        f"WHERE {' AND '.join(clauses)}"
    )
    return sql, params, idx


def _build_legislation_lookup_query(
    refs: list[tuple[str, Optional[str]]],
    match: Literal["any", "all"],
    filters: Optional[Filters],
    limit: int,
    offset: int,
) -> tuple[str, list[Any]]:
    """Build fast legislation search SQL using the normalized citation table.

    The CTE first resolves matching doc_ids from the narrow lookup table, ranks
    ids with filters applied, then hydrates full document metadata only for the
    requested page. That avoids TOAST-heavy reads of documents.metadata for
    every document citing popular codes.
    """
    if not refs:
        raise ValueError("refs must not be empty")

    params: list[Any] = []
    next_idx = 1
    selects: list[str] = []
    for match_key, (law, article) in enumerate(refs, start=1):
        select_sql, select_params, next_idx = _citation_ref_select(
            law, article, match_key, next_idx, filters
        )
        selects.append(select_sql)
        params.extend(select_params)

    if match == "all":
        matched_doc_ids = (
            "SELECT doc_id, MAX(decision_date) AS decision_date FROM matched "
            f"GROUP BY doc_id HAVING COUNT(DISTINCT match_key) = {len(refs)}"
        )
    else:
        matched_doc_ids = (
            "SELECT doc_id, MAX(decision_date) AS decision_date FROM matched "
            "GROUP BY doc_id"
        )

    doc_filter_sql, doc_filter_params, next_idx = _build_document_json_filters(
        filters, next_idx, alias="d"
    )
    params.extend(doc_filter_params)
    limit_idx = next_idx
    offset_idx = next_idx + 1
    params.extend([limit, offset])
    if doc_filter_sql:
        ranked_source = (
            "  SELECT m.doc_id, m.decision_date "
            "  FROM matched_doc_ids m "
            "  JOIN documents d ON d.doc_id = m.doc_id "
            f"  WHERE {doc_filter_sql} AND d.duplicate_of IS NULL "
        )
    else:
        ranked_source = (
            "  SELECT m.doc_id, m.decision_date FROM matched_doc_ids m "
            f"  WHERE m.doc_id {NOT_DUPLICATE_DOC_IDS} "
        )
    sql = (
        "WITH matched AS ("
        + " UNION ALL ".join(selects)
        + "), matched_doc_ids AS ("
        + matched_doc_ids
        + "), ranked_doc_ids AS ("
        + ranked_source
        + "  ORDER BY m.decision_date DESC NULLS LAST "
        f"  LIMIT ${limit_idx} OFFSET ${offset_idx}"
        ") "
        f"SELECT {_qualified_doc_columns('d')} "
        "FROM ranked_doc_ids r "
        "JOIN documents d ON d.doc_id = r.doc_id "
        "ORDER BY r.decision_date DESC NULLS LAST"
    )
    return sql, params


def _build_legislation_legacy_query(
    refs: list[tuple[str, Optional[str]]],
    match: Literal["any", "all"],
    filters: Optional[Filters],
    limit: int,
    offset: int,
) -> tuple[str, list[Any]]:
    """Build the pre-migration JSONB legislation search SQL as a fallback."""
    filt_sql, filt_params = _build_filters(filters, start_idx=1)
    all_params: list[Any] = list(filt_params)
    next_idx = len(filt_params) + 1
    conditions: list[str] = []

    for law, article in refs:
        cond_sql, cond_params, next_idx = _article_condition(law, article, next_idx)
        conditions.append(cond_sql)
        all_params.extend(cond_params)

    joiner = " OR " if match == "any" else " AND "
    where_parts = [p for p in (filt_sql, "(" + joiner.join(conditions) + ")") if p]
    limit_idx = next_idx
    offset_idx = next_idx + 1
    all_params.extend([limit, offset])

    sql = (
        f"SELECT {DOC_COLUMNS} FROM documents "
        f"WHERE {' AND '.join(where_parts)} "
        f"ORDER BY decision_date DESC NULLS LAST "
        f"LIMIT ${limit_idx} OFFSET ${offset_idx}"
    )
    return sql, all_params


async def _fetch_legislation_documents(
    refs: list[tuple[str, Optional[str]]],
    match: Literal["any", "all"],
    filters: Optional[Filters],
    limit: int,
    offset: int,
    timeout: float = 50,
) -> list[asyncpg.Record]:
    """Fetch docs via normalized citations table, falling back before migration.

    todo: remove the legacy fallback after the lookup table is deployed and
    verified in production.
    """
    lookup_sql, lookup_params = _build_legislation_lookup_query(
        refs, match, filters, limit, offset
    )
    legacy_sql, legacy_params = _build_legislation_legacy_query(
        refs, match, filters, limit, offset
    )
    async with db_pool.acquire() as conn:
        try:
            return await conn.fetch(lookup_sql, *lookup_params, timeout=timeout)
        except asyncpg.UndefinedTableError:
            logger.warning(
                "%s missing; falling back to slow JSONB legislation search",
                LEGISLATION_CITATIONS_TABLE,
            )
            return await conn.fetch(legacy_sql, *legacy_params, timeout=timeout)


class ArticleRef(BaseModel):
    """Structured (law, article) article reference."""
    law: str = Field(
        description=(
            "Law name or abbreviation (`CT`, `CPC`, `CC`, `CP`, `CPP`, …) or "
            "full canonical Portuguese name (`Código do Trabalho`, …), or "
            "statute form (`Lei n.º 65/2003`, `DL 15/93`). "
            "Resolved to the canonical full name automatically."
        ),
        examples=["CT", "Código do Trabalho", "CPC"],
    )
    article: Optional[str] = Field(
        None,
        description=(
            "Article reference (e.g. `394`, `394.º`, `394.º, n.º 2`). "
            "Prefix-matched: `394` matches `394.º`, `394.º, n.º 1`, "
            "`394.º, n.º 2, alínea b)`, etc. "
            "Omit to match any article of this law."
        ),
        examples=["394", "394.º, n.º 2"],
    )


class NormalizedArticle(BaseModel):
    """Parsed and canonicalized article reference (echoed in the response)."""
    raw: str = Field(description="Original input string.")
    law: str = Field(description="Canonical Portuguese law name.")
    article: Optional[str] = Field(
        None,
        description=(
            "Normalized article prefix used for DB LIKE matching. "
            "`null` means all articles of this law are matched."
        ),
    )
    llm_canonicalized: Optional[str] = Field(
        None,
        description="Raw LLM-normalized citation string before deterministic parsing.",
    )


class LegislationSearchRequest(BaseModel):
    """Request body for `POST /search/legislation`."""
    articles: list[Union[ArticleRef, str]] = Field(
        description=(
            "One or more article references. Each entry may be:\n\n"
            "* A **raw string** (parsed automatically): `'CT art. 394'`, "
            "`'artigo 394.º do Código do Trabalho'`, `'CPC 640'`, "
            "`'Lei n.º 65/2003 art. 5'`, `'DL 15/93 art. 26'`.\n"
            "* A **structured object** `{law, article}` for unambiguous input.\n\n"
            "Law names are resolved to canonical forms; article numbers are "
            "prefix-matched so `394` matches `394.º`, `394.º, n.º 1`, "
            "`394.º, n.º 2, alínea b)`, etc."
        ),
        examples=[["CT art. 394", "CPC art. 640"]],
    )
    match: Literal["any", "all"] = Field(
        "any",
        description=(
            "`any` (default) — OR: documents citing at least one listed article. "
            "`all` — AND: documents that cite every listed article."
        ),
    )
    limit: int = Field(20, ge=1, le=100, description="Maximum results to return.")
    offset: int = Field(0, ge=0, description="Result offset for pagination.")
    filters: Optional[Filters] = Field(
        None, description="Optional structured filters (same as `POST /search`).",
    )
    courts: Optional[list[str]] = Field(
        None,
        description="Shorthand for `filters.court` (ANY-of court codes).",
        examples=[["STJ", "TRP"]],
    )
    from_date: Optional[date] = Field(
        None,
        description=(
            "Shorthand for `filters.date_from`. "
            "Accepts `YYYY-MM-DD`, `DD/MM/YYYY`, and other formats."
        ),
    )
    to_date: Optional[date] = Field(
        None,
        description=(
            "Shorthand for `filters.date_to`. "
            "Accepts `YYYY-MM-DD`, `DD/MM/YYYY`, and other formats."
        ),
    )
    is_auj: Optional[bool] = Field(
        None, description="Shorthand for `filters.is_auj`.",
    )

    @field_validator("from_date", "to_date", mode="before")
    @classmethod
    def _coerce_date(cls, v: object) -> object:
        return _parse_flexible_date(v)

    def resolved_filters(self) -> Optional[Filters]:
        base = self.filters or Filters()
        merged = base.model_copy(update={
            "court": self.courts if self.courts is not None else base.court,
            "date_from": self.from_date if self.from_date is not None else base.date_from,
            "date_to": self.to_date if self.to_date is not None else base.date_to,
            "is_auj": self.is_auj if self.is_auj is not None else base.is_auj,
        })
        if all(v is None for v in merged.model_dump().values()):
            return None
        return merged


class LegislationSearchResponse(BaseModel):
    """Response from `POST /search/legislation`."""
    count: int = Field(description="Number of results returned (≤ `limit`).")
    offset: int = Field(description="Pagination offset used.")
    articles_searched: list[NormalizedArticle] = Field(
        description="Resolved article references echoed for traceability.",
    )
    match: str = Field(description="Match mode used: `any` or `all`.")
    filters: Optional[Filters] = Field(None, description="Filters as received.")
    results: list[SearchResult] = Field(
        description="Matching documents ordered by substantive relevance.",
    )


class LegislationSemanticSearchRequest(BaseModel):
    """Request body for `POST /search/legislation/semantic`."""
    q: str = Field(description="Free-form query describing the legislation being searched for.")
    top_k: int = Field(
        20, ge=1, le=100,
        description="Number of legislation candidates to retrieve via embedding similarity.",
    )
    limit: int = Field(20, ge=1, le=100, description="Maximum documents to return.")
    offset: int = Field(0, ge=0, description="Result offset for pagination.")
    filters: Optional[Filters] = Field(None, description="Optional structured filters.")
    courts: Optional[list[str]] = Field(
        None, description="Shorthand for `filters.court`.",
    )
    from_date: Optional[date] = Field(None, description="Shorthand for `filters.date_from`.")
    to_date: Optional[date] = Field(None, description="Shorthand for `filters.date_to`.")
    is_auj: Optional[bool] = Field(None, description="Shorthand for `filters.is_auj`.")

    @field_validator("from_date", "to_date", mode="before")
    @classmethod
    def _coerce_date(cls, v: object) -> object:
        return _parse_flexible_date(v)

    def resolved_filters(self) -> Optional[Filters]:
        base = self.filters or Filters()
        merged = base.model_copy(update={
            "court": self.courts if self.courts is not None else base.court,
            "date_from": self.from_date if self.from_date is not None else base.date_from,
            "date_to": self.to_date if self.to_date is not None else base.date_to,
            "is_auj": self.is_auj if self.is_auj is not None else base.is_auj,
        })
        if all(v is None for v in merged.model_dump().values()):
            return None
        return merged


class LegislationSemanticCandidate(BaseModel):
    """A legislation candidate retrieved via embedding similarity."""
    law: str
    article: Optional[str]
    citation_text: str
    doc_count: int
    sim: float
    selected: bool = Field(False, description="Whether the LLM deemed this relevant.")
    reason: Optional[str] = Field(None, description="LLM explanation for selection.")


class LegislationSemanticSearchResponse(BaseModel):
    """Response from `POST /search/legislation/semantic`."""
    count: int
    offset: int
    query: str
    candidates: list[LegislationSemanticCandidate]
    results: list[SearchResult]


LEGISLATION_EXAMPLES: dict[str, dict[str, Any]] = {
    "any_raw_strings": {
        "summary": "Any of: CT art. 394 OR CPC art. 640",
        "description": "Raw string inputs — law is detected automatically.",
        "value": {
            "articles": ["CT art. 394", "CPC art. 640"],
            "match": "any",
            "limit": 20,
        },
    },
    "all_structured": {
        "summary": "All of: CT art. 394 AND CT art. 395",
        "description": "Structured objects for unambiguous input.",
        "value": {
            "articles": [
                {"law": "CT", "article": "394"},
                {"law": "CT", "article": "395"},
            ],
            "match": "all",
            "limit": 20,
        },
    },
    "specific_paragraph": {
        "summary": "CT art. 394 n.º 2 — specific paragraph, STJ only",
        "description": "Prefix-match down to a specific n.º.",
        "value": {
            "articles": ["CT art. 394 n.º 2"],
            "courts": ["STJ"],
            "limit": 20,
        },
    },
    "statute": {
        "summary": "Decreto-Lei n.º 15/93 art. 21",
        "description": "Statute-style reference (DL shorthand also accepted).",
        "value": {
            "articles": ["DL 15/93 art. 21"],
            "limit": 20,
        },
    },
    "law_only": {
        "summary": "All docs citing the CRP (any article)",
        "description": "Omit article to match any citation of a given law.",
        "value": {
            "articles": [{"law": "CRP"}],
            "limit": 20,
        },
    },
}


@app.post(
    "/search/legislation",
    response_model=LegislationSearchResponse,
    tags=["search"],
    summary="Search by cited legislation articles",
    description=(
        "Return decisions that cite one or more specific legislation articles.\n\n"
        "Each entry in `articles` is a **free-form string** "
        "(e.g. `CT art. 394`, `artigo 394.º do CPC`, `DL 15/93 art. 21`) "
        "or a structured `{law, article}` object. An LLM canonicalises every "
        "raw string before parsing, so messy or abbreviated input is handled well.\n\n"
        "**Parsing:**\n\n"
        "1. The **law** is identified by abbreviation (`CT`, `CPC`, `CC`, `CP`, "
        "`CPP`, `CRP`, `CPA`, `CPTA`, `CPT`, `CSC`, `CCP`, `RCP`) or full canonical "
        "name (`Código do Trabalho`, …) or statute form "
        "(`Lei n.º 65/2003`, `DL 15/93`, `Decreto-Lei 401/82`, …).\n"
        "2. The **article** number is normalised and **prefix-matched**: "
        "`394` matches `394.º`, `394.º, n.º 1`, `394.º, n.º 2, alínea b)`, etc. "
        "Omit the article to match any citation of a law.\n\n"
        "**`match` mode:**\n\n"
        "* `any` (default) — return documents citing at least one listed article.\n"
        "* `all` — return only documents that cite every listed article.\n\n"
        "The newest matching candidates are reranked by substantive relevance. "
        "Use `offset` + `limit` for pagination.\n\n"
        "**Best practices:**\n\n"
        "- Article numbers are prefix-matched — `342` and `342.º` both work.\n"
        "- Supply multiple articles in one request; use `match='any'` for OR logic "
        "and `match='all'` when every article must appear in the same decision.\n"
        "- Omit the article to retrieve all decisions citing a law. **Avoid this "
        "for high-citation codes (CC, CPC, CP, CRP) without a date or court filter** — "
        "they match hundreds of thousands of documents and will return a 504 timeout. "
        "Always include an article number for those codes.\n"
        "- Prefer the structured `{law, article}` form when you already know the "
        "canonical law name — it bypasses the LLM canonicalisation step and is faster.\n"
        "- When the law name is ambiguous or written in natural language, use the "
        "`/search/legislation/semantic` endpoint instead.\n\n"
        "**Performance note:** The GIN index on `metadata` (`jsonb_path_ops`) is used "
        "for the law-level containment check; the article prefix filter is applied on "
        "the resulting row set. Highly-cited articles on common codes may still take "
        "several seconds."
    ),
    responses={
        400: {"description": "No valid article references after parsing."},
        422: {
            "description": (
                "A raw string did not contain a recognisable law token. "
                "Include an abbreviation (CT, CPC, CC, …) or canonical name."
            )
        },
    },
)
async def search_legislation(
    req: LegislationSearchRequest = Body(
        ..., openapi_examples=LEGISLATION_EXAMPLES
    ),
):
    # 1. Parse and normalise every article reference.
    normalized: list[NormalizedArticle] = []
    for entry in req.articles:
        if isinstance(entry, str):
            llm_canonical: Optional[str] = None
            # Always LLM-normalize raw strings first for maximum accuracy
            llm_canonical = await _llm_canonicalize_legislation_query(entry)
            if llm_canonical:
                law, article = _parse_legislation_ref(llm_canonical)
            else:
                # Safety-net: fall back to deterministic parser if LLM fails
                law, article = _parse_legislation_ref(entry)
            if not law:
                raise HTTPException(
                    422,
                    f"Could not identify a law in: {entry!r}. "
                    "Include an abbreviation (CT, CPC, CC, …) or full canonical name.",
                )
            normalized.append(
                NormalizedArticle(
                    raw=entry,
                    law=law,
                    article=article,
                    llm_canonicalized=llm_canonical,
                )
            )
        else:  # ArticleRef
            law = _canonicalize_law(entry.law)
            art: Optional[str] = None
            if entry.article:
                art = _normalize_article_prefix(entry.article)
            normalized.append(
                NormalizedArticle(
                    raw=f"{entry.law} {entry.article or ''}".strip(),
                    law=law,
                    article=art,
                )
            )

    if not normalized:
        raise HTTPException(400, "No valid article references provided.")

    effective_filters = req.resolved_filters()
    refs = [(ref.law, ref.article) for ref in normalized]
    candidate_limit = max(req.limit, RERANK_CANDIDATES) if req.offset == 0 else req.limit
    try:
        rows = await _fetch_legislation_documents(
            refs, req.match, effective_filters, candidate_limit, req.offset, timeout=50
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            504,
            "Database query timed out. The law requested is cited in many documents; "
            "try adding filters (courts, date range) or narrowing the article.",
        ) from exc

    query = "Aplicação direta em jurisprudência de " + (
        " e " if req.match == "all" else " ou "
    ).join(f"{law} {article or ''}".strip() for law, article in refs)
    ranked_docs = await _rerank_legislation_rows(query, rows)
    results = [SearchResult(**doc) for doc in ranked_docs[:req.limit]]

    return LegislationSearchResponse(
        count=len(results),
        offset=req.offset,
        articles_searched=normalized,
        match=req.match,
        filters=effective_filters,
        results=results,
    )


@app.post(
    "/search/legislation/semantic",
    response_model=LegislationSemanticSearchResponse,
    tags=["search"],
    summary="Semantic search by cited legislation (embedding + LLM rerank)",
    description=(
        "Return decisions that cite legislation semantically similar to the query.\n\n"
        "**Pipeline:**\n\n"
        "1. The query is resolved to likely Portuguese provisions using constrained "
        "structured output.\n"
        "2. Citation embeddings provide additional candidate references.\n"
        "3. Documents citing the selected `(law, article)` pairs are fetched through "
        "the normalized citation index.\n"
        "4. Matching decisions are reranked by substantive relevance to the query.\n\n"
        "**Best practices:**\n\n"
        "- Write `q` as natural Portuguese — abbreviations, full names, and mixed "
        "formats all work (e.g. `codigo civil artigo 342`, `CT art. 394`, "
        "`artigo 2 do codigo da estrada`).\n"
        "- Include the article number whenever possible. Law-only queries return "
        "results but may resolve to a high-level article (e.g. `art. 1.º`) and miss "
        "the article you intended.\n"
        "- **Avoid law-only queries for the most heavily-cited codes** (Código Civil, "
        "Código de Processo Civil, Código Penal, Constituição da República Portuguesa) "
        "without an article number — the document fetch step can time out (504) "
        "if the chosen article matches more than ~50 000 documents.\n"
        "- Use this endpoint when the law name is ambiguous or you don't know the "
        "exact canonical name. Use `POST /search/legislation` when you already have "
        "a precise citation — it is faster because it skips the embedding step.\n"
        "- Increase `top_k` (default 20) if results are missing expected legislation; "
        "the LLM sees more candidates and can make better selections.\n"
        "- `limit` controls how many matching documents are returned (not candidates)."
    ),
)
async def search_legislation_semantic(
    req: LegislationSemanticSearchRequest = Body(...),
):
    if not req.q or not req.q.strip():
        raise HTTPException(400, "Query parameter 'q' is required.")

    # 1. Embed the query
    try:
        emb = await embed_query(req.q.strip())
    except Exception as exc:
        raise HTTPException(502, f"Embedding API error: {exc}") from exc

    emb_str = "[" + ",".join(str(v) for v in emb) + "]"

    # 2. Vector search legislation_embeddings
    async with db_pool.acquire() as conn:
        leg_rows = await conn.fetch(
            """
            SELECT law, article, citation_text, doc_count,
                   (1 - (embedding <=> $1::halfvec))::real AS sim
            FROM legislation_embeddings
            ORDER BY embedding <=> $1::halfvec
            LIMIT $2
            """,
            emb_str, req.top_k,
        )

    if not leg_rows:
        return LegislationSemanticSearchResponse(
            count=0, offset=req.offset, query=req.q,
            candidates=[], results=[],
        )

    candidates = [
        {"idx": i, "law": r["law"], "article": r["article"],
         "citation_text": r["citation_text"], "doc_count": r["doc_count"],
         "sim": float(r["sim"])}
        for i, r in enumerate(leg_rows)
    ]

    # 3. Resolve the legal intent directly. Citation-only embeddings cannot
    # reliably infer an article from a conceptual description.
    resolved = await _resolve_legislation_intent(req.q)
    selected = []
    for ref in resolved:
        idx = next(
            (
                i for i, candidate in enumerate(candidates)
                if candidate["law"] == ref["law"]
                and candidate["article"] == ref["article"]
            ),
            None,
        )
        if idx is None:
            idx = len(candidates)
            candidates.append({
                "idx": idx,
                "law": ref["law"],
                "article": ref["article"],
                "citation_text": f"{ref['law']} art. {ref['article']}",
                "doc_count": 0,
                "sim": 0.0,
            })
        selected.append({"idx": idx, "reason": ref["reason"]})
    if not selected:
        selected = await _llm_rerank_legislation(req.q, candidates)
    selected_indices = {s["idx"] for s in selected}

    candidate_models = [
        LegislationSemanticCandidate(
            law=c["law"],
            article=c["article"],
            citation_text=c["citation_text"],
            doc_count=c["doc_count"],
            sim=c["sim"],
            selected=i in selected_indices,
            reason=next((s["reason"] for s in selected if s["idx"] == i), None),
        )
        for i, c in enumerate(candidates)
    ]

    if not selected_indices:
        return LegislationSemanticSearchResponse(
            count=0,
            offset=req.offset,
            query=req.q,
            candidates=candidate_models,
            results=[],
        )

    # 4. Fetch documents through the normalized citation lookup table.
    effective_filters = req.resolved_filters()
    selected_refs = [
        (candidates[idx]["law"], candidates[idx]["article"])
        for idx in sorted(selected_indices)
    ]
    candidate_limit = max(req.limit, RERANK_CANDIDATES) if req.offset == 0 else req.limit
    try:
        rows = await _fetch_legislation_documents(
            selected_refs, "any", effective_filters, candidate_limit, req.offset, timeout=50
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            504,
            "Database query timed out. The selected legislation is cited in many documents; "
            "try adding filters (courts, date range) or narrowing the query.",
        ) from exc

    ranked_docs = await _rerank_legislation_rows(req.q, rows)
    results = [SearchResult(**doc) for doc in ranked_docs[:req.limit]]

    return LegislationSemanticSearchResponse(
        count=len(results),
        offset=req.offset,
        query=req.q,
        candidates=candidate_models,
        results=results,
    )
