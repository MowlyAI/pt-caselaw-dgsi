"""MCP server for PT Caselaw DGSI — exposes search as MCP tools via fastmcp.

Run standalone (stdio, for Claude Desktop / Claude Code local):
    python -m api.mcp_server

Run standalone (HTTP on a separate port):
    fastmcp run api/mcp_server.py --transport http --port 8001

Run alongside the FastAPI app (shared db pool, single process):
    uvicorn api.app:app        # MCP endpoint at /mcp/
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal, Optional

import asyncpg
import httpx
from fastmcp import FastMCP

import api.main as _m


@asynccontextmanager
async def _lifespan(server: FastMCP):
    """Initialise db pool and HTTP client when running standalone.
    When mounted under FastAPI, the pool is already live — skipped.
    """
    owns = _m.db_pool is None
    if owns:
        from dotenv import load_dotenv
        load_dotenv(".env.local")
        _m.http_client = httpx.AsyncClient(timeout=30)
        _m.db_pool = await asyncpg.create_pool(
            host=_m.DB_HOST, port=_m.DB_PORT,
            user=_m.DB_USER, password=_m.DB_PASSWORD,
            database=_m.DB_NAME,
            min_size=1, max_size=5,
            statement_cache_size=0,
            command_timeout=30,
            init=_m._init_connection,
        )
    yield
    if owns:
        await _m.db_pool.close()
        await _m.http_client.aclose()
        _m.db_pool = None
        _m.http_client = None


mcp = FastMCP(
    "PT Caselaw DGSI",
    instructions=(
        "Search over Portuguese court decisions from DGSI "
        "(STJ, STA, TC, TRP, TRL, TRC, TRG, TCAN, TCAS and others).\n\n"
        "Recommended workflow:\n"
        "1. Call get_filters to discover available courts and the date range.\n"
        "2. Call search with a natural-language query and optional filters.\n"
        "3. Call get_document on interesting results to get full metadata "
        "(parties, legal citations, ratio decidendi, amounts, timeline)."
    ),
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_filters(
    court: Optional[list[str]],
    legal_domain: Optional[str],
    is_auj: Optional[bool],
    date_from: Optional[str],
    date_to: Optional[str],
    decision_type: Optional[list[str]],
) -> Optional[_m.Filters]:
    if not any([court, legal_domain, is_auj is not None,
                date_from, date_to, decision_type]):
        return None
    return _m.Filters(
        court=court,
        legal_domain=legal_domain,
        is_auj=is_auj,
        date_from=date.fromisoformat(date_from) if date_from else None,
        date_to=date.fromisoformat(date_to) if date_to else None,
        decision_type=decision_type,
    )


def _serialise(results: list[_m.SearchResult]) -> list[dict]:
    out = []
    for r in results:
        d = r.model_dump(exclude_none=True)
        if isinstance(d.get("decision_date"), date):
            d["decision_date"] = d["decision_date"].isoformat()
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool
async def search(
    q: str,
    limit: int = 10,
    mode: Literal["hybrid", "semantic", "fts"] = "hybrid",
    court: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    legal_domain: Optional[str] = None,
    is_auj: Optional[bool] = None,
    decision_type: Optional[list[str]] = None,
) -> list[dict]:
    """Search Portuguese court decisions (DGSI).

    Args:
        q: Query string. Natural language for hybrid/semantic (e.g. "despedimento
           sem justa causa por uso indevido de email"). Keywords with optional
           operators for fts (e.g. '"acidente de viação" -trabalho').
        limit: Max results to return (1–50, default 10).
        mode: Search strategy — "hybrid" (3 vector columns + FTS fused with RRF,
              recommended), "semantic" (vectors only, good for conceptual queries),
              "fts" (keyword-only, no embedding call, lowest latency, good for exact
              terms, process numbers, statute references).
        court: Restrict to one or more court codes e.g. ["STJ", "TRP"].
               Call get_filters to see all available codes with document counts.
        date_from: Earliest decision date, ISO format YYYY-MM-DD (inclusive).
        date_to: Latest decision date, ISO format YYYY-MM-DD (inclusive).
        legal_domain: Case-insensitive substring matched against the legal_domain
                      field (e.g. "insolvencia"). High-cardinality — use a substring.
        is_auj: true = only binding precedents (Acórdão de Uniformização /
                Fixação de Jurisprudência); false = exclude binding precedents.
        decision_type: Filter by decision type e.g. ["Acórdão", "Sentença", "Despacho"].
    """
    limit = max(1, min(limit, 50))
    filters = _make_filters(court, legal_domain, is_auj, date_from, date_to, decision_type)
    overfetch = limit * 4
    weights_obj = _m.SearchWeights()

    if mode == "fts":
        per_source: dict[str, list] = {"fts": await _m._search_fts(q, overfetch, filters)}
        weight_map: dict[str, float] = {"fts": 1.0}
    elif mode == "semantic":
        vec_fields = _m._enabled_vector_fields(weights_obj)
        per_source = await _m._search_vectors(q, vec_fields, overfetch, filters)
        weight_map = {f: 1.0 for f in vec_fields}
    else:  # hybrid
        vec_fields = _m._enabled_vector_fields(weights_obj)
        vec_res, fts_hits = await asyncio.gather(
            _m._search_vectors(q, vec_fields, overfetch, filters),
            _m._search_fts(q, overfetch, filters),
        )
        per_source = {**vec_res, "fts": fts_hits}
        weight_map = {f: 1.0 for f in [*vec_fields, "fts"]}

    merged, ranks = _m._rrf_merge_multi(per_source, weight_map)
    docs = await _m._fetch_docs([doc_id for doc_id, _ in merged[:limit]])
    results = _m._build_results(
        merged, docs, per_source, ranks, limit,
        include_hybrid=(mode == "hybrid"),
    )
    return _serialise(results)


@mcp.tool
async def get_document(doc_id: str) -> dict:
    """Fetch the full record for a single court decision by its doc_id.

    Returns the complete metadata blob: parties, legal citations, ratio decidendi,
    monetary amounts, procedural timeline, extraction confidence, and more.
    Use this after search to hydrate a result you want to read in full.

    Args:
        doc_id: Stable document identifier returned by search results.
    """
    docs = await _m._fetch_docs([doc_id])
    if doc_id not in docs:
        raise ValueError(f"Document not found: {doc_id!r}")
    doc = docs[doc_id]
    if isinstance(doc.get("decision_date"), date):
        doc["decision_date"] = doc["decision_date"].isoformat()
    return doc


@mcp.tool
async def get_filters() -> dict:
    """Discover available filter values for the search tool.

    Returns every court code with document counts, the overall date range,
    the 50 most common legal domains, decision types, and extraction confidence
    levels. Call once at the start of a session to ground your filter choices.
    """
    return await _m._compute_filters_payload(None, 50)
