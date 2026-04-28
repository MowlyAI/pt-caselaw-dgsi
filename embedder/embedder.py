"""
Embedder: generates embeddings via OpenRouter and builds Supabase rows.
Uses qwen/qwen3-embedding-8b (1024 dimensions, 32K context).

Produces 3 independent embeddings per document:
  - embedding:         semantic_search_query (LLM-optimized retrieval query)
  - embedding_context: semantic_search_query + legal_question + summary
  - embedding_ratio:   ratio_decidendi + decision_outcome

All metadata comes from `doc["llm_extracted"]` — the scraper only provides full_text.
"""
from __future__ import annotations

import asyncio
import os
import re
from datetime import date
from typing import Optional

import httpx

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

# Qwen3 supports 32K tokens; we set a generous char limit.
MAX_EMBEDDING_CHARS = 12_000


def build_embedding_texts(doc: dict) -> dict[str, str]:
    """Build 3 independent embedding inputs from the LLM's focused fields.

    Returns a dict with keys matching the DB column names:
      - embedding:         semantic_search_query alone
      - embedding_context: semantic_search_query + legal_question + summary
      - embedding_ratio:   ratio_decidendi + decision_outcome
    """
    llm = doc.get("llm_extracted") or {}
    ssq = llm.get("semantic_search_query", "")
    lq = llm.get("legal_question", "")
    summary = llm.get("summary", "")
    ratio = llm.get("ratio_decidendi", "")
    outcome = llm.get("decision_outcome", "")

    texts = {}

    # 1. Primary: semantic_search_query alone
    texts["embedding"] = (ssq or summary or "")[:MAX_EMBEDDING_CHARS]

    # 2. Context: broader semantic context
    ctx_parts = [p for p in [ssq, lq, summary] if p]
    texts["embedding_context"] = ("\n\n".join(ctx_parts) or "")[:MAX_EMBEDDING_CHARS]

    # 3. Ratio: legal reasoning
    ratio_parts = [p for p in [ratio, outcome] if p]
    texts["embedding_ratio"] = ("\n\n".join(ratio_parts) or "")[:MAX_EMBEDDING_CHARS]

    return texts


async def generate_embedding(
    client: httpx.AsyncClient,
    text: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> Optional[list[float]]:
    """Generate a single embedding via OpenRouter."""
    result = await generate_embeddings_batch(client, [text], api_key, semaphore, max_retries)
    return result[0] if result else None


# Texts per batch request — keep moderate to avoid timeouts on large payloads
BATCH_SIZE = 50


async def generate_embeddings_batch(
    client: httpx.AsyncClient,
    texts: list[str],
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> Optional[list[Optional[list[float]]]]:
    """Generate embeddings for a batch of texts in a single API call.

    Returns a list of embeddings (same order as input texts).
    Returns None only on total failure.
    """
    if not texts:
        return []
    async with semaphore:
        payload = {"model": EMBEDDING_MODEL, "input": texts, "dimensions": EMBEDDING_DIM}
        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    f"{OPENROUTER_BASE}/embeddings",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/pt-caselaw-dgsi",
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()["data"]
                    # Sort by index to maintain input order
                    data.sort(key=lambda x: x["index"])
                    return [d["embedding"] for d in data]
                if resp.status_code in (429, 500, 502, 503, 504, 529):
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
            except (httpx.TimeoutException, httpx.RequestError):
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
        return None


_DATE_PATTERNS = [
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"^(\d{2})/(\d{2})/(\d{4})$"), lambda m: date(int(m[3]), int(m[1]), int(m[2]))),
    (re.compile(r"^(\d{2})-(\d{2})-(\d{4})$"), lambda m: date(int(m[3]), int(m[2]), int(m[1]))),
]


def parse_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    for pattern, convert in _DATE_PATTERNS:
        m = pattern.match(s)
        if m:
            try:
                return convert(m).isoformat()
            except ValueError:
                return None
    return None


def doc_to_row(doc: dict) -> dict:
    """Convert a document to a simplified Supabase row.

    Filter columns are top-level; everything else goes into metadata JSONB.
    """
    llm = doc.get("llm_extracted") or {}

    # Build metadata: the entire llm_extracted dict + source info
    metadata = dict(llm)
    metadata["source_db"] = doc.get("source_db", "")
    metadata["court"] = doc.get("court", "")
    metadata["court_short"] = doc.get("court_short", "")

    return {
        "doc_id": doc.get("doc_id", ""),
        "url": doc.get("url", ""),
        "court_short": doc.get("court_short", ""),
        "decision_date": parse_date(llm.get("decision_date")),
        "process_number": llm.get("process_number"),
        "legal_domain": llm.get("legal_domain"),
        "is_auj": bool(llm.get("is_jurisprudence_unification")),
        "full_text": doc.get("full_text", ""),
        "summary": llm.get("summary"),
        "metadata": metadata,
    }
