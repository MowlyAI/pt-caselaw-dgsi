"""MCP server for PT Caselaw DGSI — exposes search as MCP tools via fastmcp.

## Claude Desktop (stdio — recommended for local use)
Add this block to ~/Library/Application Support/Claude/claude_desktop_config.json:

    {
      "mcpServers": {
        "pt-caselaw-dgsi": {
          "command": "/Users/franciscocosta/repos/pt-caselaw-dgsi/.venv312/bin/python3.12",
          "args": ["-m", "api.mcp_server"],
          "env": {
            "PYTHONPATH": "/Users/franciscocosta/repos/pt-caselaw-dgsi"
          }
        }
      }
    }

PYTHONPATH is required because Claude Desktop launches the process from its own
working directory, not the project root, so the `api` package would not be found.
Credentials are loaded from .env.local via an absolute path (no cwd dependency).

## Remote HTTP (Claude.ai / Anthropic API connector)
Start the combined FastAPI + MCP server:
    uvicorn api.app:app --host 0.0.0.0 --port 8000

Streamable HTTP endpoint (Claude.ai remote MCP): http://localhost:8000/mcp/
SSE endpoint (legacy Claude Desktop HTTP mode):  http://localhost:8000/sse

## Standalone HTTP (separate process)
    fastmcp run api/mcp_server.py:mcp --transport http --port 8001
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

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
        # api.main already calls load_dotenv at import time with an absolute path.
        # Nothing extra needed here.
        _m.http_client = httpx.AsyncClient(timeout=30)
        _m.db_pool = await asyncpg.create_pool(
            host=_m.DB_HOST, port=_m.DB_PORT,
            user=_m.DB_USER, password=_m.DB_PASSWORD,
            database=_m.DB_NAME,
            min_size=0, max_size=5,  # min_size=0: don't block startup with a network round-trip
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

        "## Available tools\n"
        "- get_filters — discover court codes and date range (call once per session)\n"
        "- search — hybrid semantic + keyword search over all decisions\n"
        "- search_by_legislation — find decisions that cite specific laws/articles\n"
        "- get_document — fetch the full record for a single decision\n\n"

        "## Recommended workflow\n"
        "1. Call get_filters to discover available court codes and the corpus date range.\n"
        "2. Use search for open-ended questions about legal doctrine, facts, or outcomes.\n"
        "   Always supply BOTH q_semantic (the full natural-language question) AND "
        "q_keywords (important Portuguese terms). This drives both the vector and FTS "
        "engines and produces the best-ranked results.\n"
        "3. Use search_by_legislation when the user cites a specific law or article "
        "(e.g. 'artigo 342.º do Código Civil', 'CT art. 394'). Pass as many article "
        "references as needed; use match='all' if every article must be cited.\n"
        "4. Call get_document on interesting results to read parties, legal citations, "
        "ratio decidendi, amounts, and the full procedural timeline.\n\n"

        "## Best practices — general search (search tool)\n"
        "- Write q_semantic as a complete legal question or description, not a keyword list: "
        "e.g. 'despedimento sem justa causa por uso indevido de email corporativo' "
        "beats 'despedimento email'.\n"
        "- Extract 3–6 discriminating Portuguese nouns/verbs for q_keywords. "
        "These feed a GIN full-text index (Portuguese stemming + unaccent); "
        "they do NOT need to match q_semantic word-for-word.\n"
        "- Apply court and date filters to narrow results before increasing limit.\n"
        "- Set is_auj=true to retrieve only binding precedents (Acórdãos de "
        "Uniformização de Jurisprudência / Fixação de Jurisprudência).\n"
        "- Use weights to bias the ranking: boost embedding_ratio (legal reasoning) "
        "for doctrinal questions; boost embedding_context (facts + parties + decision) "
        "for factual similarity searches.\n\n"

        "## Best practices — legislation search (search_by_legislation tool)\n"
        "- Pass article references exactly as the user wrote them; the parser handles "
        "Portuguese abbreviations (CT, CC, CPC, CP, CPP, CRP, CPA, CPTA, CSC, CCP, "
        "RCP) and statute forms (DL 15/93 art. 21, Lei n.º 65/2003 art. 3).\n"
        "- Article numbers are prefix-matched: '394' matches '394.º', '394.º, n.º 1', "
        "'394.º, n.º 2, alínea b)', etc. You do NOT need to include the ordinal suffix.\n"
        "- Omit the article number to match any article of a law (e.g. 'CRP' returns "
        "all decisions citing the Constitution).\n"
        "- Use match='any' (default) for OR logic and match='all' for AND logic when "
        "multiple articles are supplied.\n"
        "- Avoid law-only queries on the most heavily-cited codes (CC, CPC, CP, CRP) "
        "without a date or court filter — they match hundreds of thousands of documents "
        "and will time out. Always include an article number for those codes.\n"
        "- Use pagination (offset) to page through large result sets.\n"
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
    q_semantic: str,
    q_keywords: list[str],
    limit: int = 10,
    court: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    legal_domain: Optional[str] = None,
    is_auj: Optional[bool] = None,
    decision_type: Optional[list[str]] = None,
) -> list[dict]:
    """Search Portuguese court decisions (DGSI) using hybrid search (3 vector columns + FTS fused with RRF).

    Best practices:
    - Always supply BOTH q_semantic and q_keywords for the best results.
      q_semantic drives the three HNSW vector indexes (summary, context, ratio);
      q_keywords drives the GIN full-text index (Portuguese stemming + unaccent).
      Fusing all four sources with Reciprocal Rank Fusion yields the highest recall.
    - Write q_semantic as a complete legal sentence, not a keyword list.
      Good: "despedimento sem justa causa por uso indevido de email corporativo"
      Bad:  "despedimento email"
    - For q_keywords extract 3–6 discriminating Portuguese nouns or verbs.
      They do NOT need to mirror q_semantic word-for-word.
    - Filter first, then increase limit: apply court + date filters before
      widening the result window.
    - Set is_auj=true to retrieve only binding uniformisation precedents.
    - For doctrinal questions weight embedding_ratio higher (legal reasoning column).
      For factual similarity weight embedding_context higher (facts + outcome column).

    Args:
        q_semantic: Natural-language query for vector search (e.g. "despedimento
                    sem justa causa por uso indevido de email corporativo").
        q_keywords: List of keywords for full-text search (e.g.
                    ["despedimento", "email", "corporativo"]). Supports
                    `websearch_to_tsquery` syntax: quoted "phrase", -excluded, OR.
        limit: Max results to return (1–50, default 10).
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
    vec_fields = _m._enabled_vector_fields(weights_obj)
    kw_q = " ".join(q_keywords)

    vec_res, fts_hits = await asyncio.gather(
        _m._search_vectors(q_semantic, vec_fields, overfetch, filters),
        _m._search_fts(kw_q, overfetch, filters),
    )
    per_source: dict[str, list] = {**vec_res, "fts": fts_hits}
    weight_map: dict[str, float] = {f: 1.0 for f in [*vec_fields, "fts"]}

    merged, ranks = _m._rrf_merge_multi(per_source, weight_map)
    docs = await _m._fetch_docs([doc_id for doc_id, _ in merged[:limit]])
    results = _m._build_results(
        merged, docs, per_source, ranks, limit, include_hybrid=True,
    )
    return _serialise(results)


@mcp.tool
async def get_document(doc_id: str, include_full_text: bool = False) -> dict:
    """Fetch the full record for a single court decision by its doc_id.

    Returns the complete metadata blob: parties, legal citations, ratio decidendi,
    monetary amounts, procedural timeline, extraction confidence, and more.
    Use this after search to hydrate a result you want to read in full.

    Args:
        doc_id: Stable document identifier returned by search results.
        include_full_text: When True, the response includes a `full_text` field
            with the complete integral text of the decision as scraped from DGSI.
            Omitted by default to keep responses compact.
    """
    docs = await _m._fetch_docs([doc_id])
    if doc_id not in docs:
        raise ValueError(f"Document not found: {doc_id!r}")
    doc = docs[doc_id]
    if isinstance(doc.get("decision_date"), date):
        doc["decision_date"] = doc["decision_date"].isoformat()
    if include_full_text:
        async with _m.db_pool.acquire() as conn:
            full_text = await conn.fetchval(
                "SELECT full_text FROM documents WHERE doc_id = $1", doc_id
            )
        doc["full_text"] = full_text
    return doc


@mcp.tool
async def search_by_legislation(
    articles: list[str],
    match: str = "any",
    limit: int = 10,
    court: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    is_auj: Optional[bool] = None,
    offset: int = 0,
) -> dict:
    """Find court decisions that cite specific legislation articles.

    Each entry in `articles` is a free-form Portuguese article reference parsed
    automatically. The law is identified by abbreviation or full name; the article
    is prefix-matched against the stored canonical form in the database.

    Best practices:
    - Pass references exactly as the user wrote them; the parser is tolerant of
      typos, missing accents, and mixed formats (e.g. 'artigo 394 do CT',
      'CT art. 394', 'Código do Trabalho 394').
    - Article numbers are prefix-matched: '394' matches '394.º', '394.º, n.º 1',
      '394.º, n.º 2, alínea b)', etc. You do NOT need the ordinal suffix.
    - Omit the article to match ALL articles of a law (e.g. 'CRP' for Constitution).
      Avoid this pattern for the most heavily-cited codes (CC, CPC, CP, CRP) without
      a court or date filter — they match hundreds of thousands of docs and will time out.
    - Use match='all' only when the user explicitly requires every cited article to be
      present in the same decision. Default 'any' is correct for most queries.
    - For multi-article queries use match='any' with a long articles list rather than
      calling the tool multiple times.
    - Page through large result sets with offset rather than increasing limit above 50.

    Known abbreviations: CT, CPC, CC, CP, CPP, CRP, CPA, CPTA, CPT, CSC, CCP, RCP
    and their full canonical Portuguese names. Statute forms 'Lei n.º X/Y',
    'Decreto-Lei n.º X/Y' (or 'DL X/Y') are also recognised.

    Args:
        articles: List of article references as free-form strings.
                  Examples: 'CT art. 394', 'artigo 394.º do CPC',
                  'Código Civil 483', 'DL 15/93 art. 21',
                  'CPC art. 640 n.º 3', 'Lei n.º 65/2003 art. 5'.
        match: 'any' (default) — documents citing at least one listed article.
               'all' — documents citing every listed article.
        limit: Max results to return (1–50, default 10).
        court: Restrict to one or more court codes e.g. ["STJ", "TRP"].
        date_from: Earliest decision date ISO YYYY-MM-DD (inclusive).
        date_to: Latest decision date ISO YYYY-MM-DD (inclusive).
        is_auj: true = only binding precedents; false = exclude them.
        offset: Result offset for pagination (default 0).
    """
    from datetime import date as _date
    limit = max(1, min(limit, 50))
    filters = _make_filters(court, None, is_auj, date_from, date_to, None)

    req = _m.LegislationSearchRequest(
        articles=articles,
        match=match,  # type: ignore[arg-type]
        limit=limit,
        offset=offset,
        filters=filters,
    )

    # Reuse the REST endpoint logic directly.
    resp = await _m.search_legislation(req)

    return {
        "articles_searched": [a.model_dump() for a in resp.articles_searched],
        "match": resp.match,
        "count": resp.count,
        "offset": resp.offset,
        "results": _serialise(resp.results),
    }


@mcp.tool
async def get_filters() -> dict:
    """Discover available courts and the corpus date range for use as search filters.

    Returns:
        courts: list of {value, count} — every court_short code with document counts.
                Pass these as the `court` filter in search (e.g. ["STJ", "TRP"]).
        decision_date: {min, max} — earliest and latest decision dates in the corpus.
                       Use as bounds for date_from / date_to in search.
        is_auj: list of {value, count} — how many decisions are binding precedents.

    Call once at the start of a session to ground filter choices.
    """
    return await _m._compute_filters_payload()


if __name__ == "__main__":
    # stdio transport — used by Claude Desktop when it launches this file as a subprocess.
    # The FastMCP lifespan initialises the DB pool and HTTP client on startup.
    # show_banner=False keeps stderr clean so Claude Desktop can parse MCP output correctly.
    mcp.run(show_banner=False)
