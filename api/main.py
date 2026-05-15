"""FastAPI app for hybrid search over DGSI Portuguese caselaw.

Uses Supabase Postgres directly via asyncpg:
  * HNSW indexes on `embedding` / `embedding_context` / `embedding_ratio`
    (halfvec_cosine_ops) for vector search.
  * GIN index on `fts` (tsvector built with unaccent + portuguese) for FTS.
"""
import asyncio
import json as _json
import os
import re
import time
import unicodedata
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, Literal, Optional, Union

import asyncpg
import httpx
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Path as FastAPIPath, Query
from pydantic import BaseModel, Field, field_validator

# Use absolute path so the server works regardless of the working directory
# (e.g. when launched as a Claude Desktop subprocess from the home directory).
_ENV_FILE = Path(__file__).parent.parent / ".env.local"
load_dotenv(_ENV_FILE)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
LLM_MODEL = os.getenv("LLM_MODEL", "xiaomi/mimo-v2-flash")

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

    @field_validator("date_from", "date_to", mode="before")
    @classmethod
    def _coerce_date(cls, v: object) -> object:
        return _parse_flexible_date(v)


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
        merged = Filters(
            court=self.courts if self.courts is not None else base.court,
            date_from=self.from_date if self.from_date is not None else base.date_from,
            date_to=self.to_date if self.to_date is not None else base.date_to,
            is_auj=self.is_auj if self.is_auj is not None else base.is_auj,
            legal_domain=base.legal_domain,
            decision_type=base.decision_type,
            extraction_confidence=base.extraction_confidence,
        )
        if all(v is None for v in [
            merged.court, merged.date_from, merged.date_to, merged.is_auj,
            merged.legal_domain, merged.decision_type, merged.extraction_confidence,
        ]):
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
        "**Best practices:**\n\n"
        "- Supply **both** `q_semantic` and `q_keywords` for maximum recall. "
        "Omitting either disables that source.\n"
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
    sem_q, kw_q = _resolve_queries(req, need_sem=bool(vec_fields), need_fts=use_fts)
    over = req.limit * req.overfetch
    effective_filters = req.resolved_filters()

    if vec_fields and use_fts:
        per_vec, fts_hits = await asyncio.gather(
            _search_vectors(sem_q, vec_fields, over, effective_filters),
            _search_fts(kw_q, over, effective_filters),
        )
    elif vec_fields:
        per_vec = await _search_vectors(sem_q, vec_fields, over, effective_filters)
        fts_hits = []
    else:
        per_vec = {}
        fts_hits = await _search_fts(kw_q, over, effective_filters)

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
    return {
        "courts": [{"value": r["value"], "count": r["count"]} for r in court_rows],
        "decision_date": {
            "min": date_row["min"].isoformat() if date_row["min"] else None,
            "max": date_row["max"].isoformat() if date_row["max"] else None,
        },
        "is_auj": [{"value": r["value"], "count": r["count"]} for r in is_auj_rows],
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
        "* `is_auj` — count of binding-precedent (AUJ) decisions vs regular ones.\n\n"
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
# Legislation article search
# ---------------------------------------------------------------------------

from extractor.extractor import _LAW_ABBREV, _canonicalize_law, _clean_article_text  # noqa: E402

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
    if not re.match(r"^\d+(?:-[A-Za-z])?\.º", s):
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
        merged = Filters(
            court=self.courts if self.courts is not None else base.court,
            date_from=self.from_date if self.from_date is not None else base.date_from,
            date_to=self.to_date if self.to_date is not None else base.date_to,
            is_auj=self.is_auj if self.is_auj is not None else base.is_auj,
            legal_domain=base.legal_domain,
            decision_type=base.decision_type,
            extraction_confidence=base.extraction_confidence,
        )
        if all(v is None for v in [
            merged.court, merged.date_from, merged.date_to, merged.is_auj,
            merged.legal_domain, merged.decision_type, merged.extraction_confidence,
        ]):
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
        description="Matching documents ordered by decision_date descending.",
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
        merged = Filters(
            court=self.courts if self.courts is not None else base.court,
            date_from=self.from_date if self.from_date is not None else base.date_from,
            date_to=self.to_date if self.to_date is not None else base.date_to,
            is_auj=self.is_auj if self.is_auj is not None else base.is_auj,
            legal_domain=base.legal_domain,
            decision_type=base.decision_type,
            extraction_confidence=base.extraction_confidence,
        )
        if all(v is None for v in [
            merged.court, merged.date_from, merged.date_to, merged.is_auj,
            merged.legal_domain, merged.decision_type, merged.extraction_confidence,
        ]):
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
        "Results are ordered by `decision_date` descending. "
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

    # 2. Build SQL WHERE clause.
    effective_filters = req.resolved_filters()
    filt_sql, filt_params = _build_filters(effective_filters, start_idx=1)

    all_params: list[Any] = list(filt_params)
    next_idx = len(filt_params) + 1
    conditions: list[str] = []

    for ref in normalized:
        cond_sql, cond_params, next_idx = _article_condition(
            ref.law, ref.article, next_idx
        )
        conditions.append(cond_sql)
        all_params.extend(cond_params)

    joiner = " OR " if req.match == "any" else " AND "
    article_clause = "(" + joiner.join(conditions) + ")"

    where_parts: list[str] = []
    if filt_sql:
        where_parts.append(filt_sql)
    where_parts.append(article_clause)
    where_clause = " AND ".join(where_parts)

    limit_idx = next_idx
    offset_idx = next_idx + 1
    all_params.extend([req.limit, req.offset])

    sql = (
        f"SELECT {DOC_COLUMNS} FROM documents "
        f"WHERE {where_clause} "
        f"ORDER BY decision_date DESC NULLS LAST "
        f"LIMIT ${limit_idx} OFFSET ${offset_idx}"
    )

    async with db_pool.acquire() as conn:
        try:
            rows = await conn.fetch(sql, *all_params, timeout=50)
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                504,
                "Database query timed out. The law requested is cited in many documents; "
                "try adding filters (courts, date range) or narrowing the article.",
            ) from exc

    results = [SearchResult(**_row_to_dict(r)) for r in rows]

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
        "1. The query is embedded (google/gemini-embedding-001, 1024 dims).\n"
        "2. Top-K candidates are retrieved from `legislation_embeddings` via HNSW "
        "vector search (cosine similarity).\n"
        "3. An LLM reranks the candidates, selecting the genuinely relevant "
        "`(law, article)` pairs and discarding false positives.\n"
        "4. Documents citing the selected legislation are fetched via a fast "
        "GIN exact-match query (`metadata @> '{\"legislation_cited\": [...]}'`).\n\n"
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

    # 3. LLM reranking
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

    # If LLM rejected all candidates, fall back to top-1 as a safety net
    if not selected_indices:
        selected_indices = {0}
        candidate_models[0].selected = True
        candidate_models[0].reason = "Fallback: top similarity match"

    # 4. Build document query for selected legislation using fast exact-match @> via GIN
    effective_filters = req.resolved_filters()
    filt_sql, filt_params = _build_filters(effective_filters, start_idx=1)
    all_params: list[Any] = list(filt_params)
    next_idx = len(filt_params) + 1
    conditions: list[str] = []

    for idx in selected_indices:
        leg = leg_rows[idx]
        law = leg["law"]
        article = leg["article"]
        match_obj: dict[str, Any] = {"legislation_cited": [{"law": law}]}
        if article and article.strip():
            match_obj["legislation_cited"][0]["article"] = article.strip()
        conditions.append(f"metadata @> ${next_idx}::jsonb")
        all_params.append(_json.dumps(match_obj))
        next_idx += 1

    article_clause = "(" + " OR ".join(conditions) + ")"

    where_parts: list[str] = []
    if filt_sql:
        where_parts.append(filt_sql)
    where_parts.append(article_clause)
    where_clause = " AND ".join(where_parts)

    limit_idx = next_idx
    offset_idx = next_idx + 1
    all_params.extend([req.limit, req.offset])

    sql = (
        f"SELECT {DOC_COLUMNS} FROM documents "
        f"WHERE {where_clause} "
        f"ORDER BY decision_date DESC NULLS LAST "
        f"LIMIT ${limit_idx} OFFSET ${offset_idx}"
    )

    async with db_pool.acquire() as conn:
        try:
            rows = await conn.fetch(sql, *all_params, timeout=50)
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                504,
                "Database query timed out. The selected legislation is cited in many documents; "
                "try adding filters (courts, date range) or narrowing the query.",
            ) from exc

    results = [SearchResult(**_row_to_dict(r)) for r in rows]

    return LegislationSemanticSearchResponse(
        count=len(results),
        offset=req.offset,
        query=req.q,
        candidates=candidate_models,
        results=results,
    )
