"""Map raw documents.legal_domain values onto canonical legal areas.

Only raw values not yet in legal_domain_map are sent to the LLM, so after the
initial run each refresh maps just the handful of new strings. Answers are
constrained to api.legal_areas.CANONICAL_AREAS by a JSON schema and validated.

Usage:
    python scripts/canonicalize_legal_domains.py
    python scripts/canonicalize_legal_domains.py --dry-run   # count only
"""
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import asyncpg
import click
import httpx
from dotenv import load_dotenv

load_dotenv(".env.local")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.legal_areas import ALIASES, CANONICAL_AREAS, normalize

MODEL = "google/gemini-3.1-flash-lite"

# Exact component names → areas. The LLM occasionally drops an area the raw
# text names explicitly (e.g. "Direito Administrativo / Direito Civil" → Civil
# only, 4k documents), so explicit names are always unioned in.
_EXPLICIT = {normalize(a): [a] for a in CANONICAL_AREAS}
_EXPLICIT.update({f"direito {alias}": areas for alias, areas in ALIASES.items()})
_EXPLICIT.update({f"direito da {alias}": areas for alias, areas in ALIASES.items()})
_EXPLICIT.update({f"direito do {alias}": areas for alias, areas in ALIASES.items()})
_EXPLICIT.update({
    "direito da familia": ["Direito da Família e Menores"],
    "direito de familia": ["Direito da Família e Menores"],
    "direito comercial": ["Direito Comercial e Societário"],
    "direito societario": ["Direito Comercial e Societário"],
    "direito bancario": ["Direito Bancário e Financeiro"],
    "direito do urbanismo": ["Direito do Urbanismo e Ordenamento do Território"],
    "direito processual administrativo": ["Direito Administrativo"],
    "direito processual tributario": ["Direito Tributário"],
    "direito processual do trabalho": ["Direito do Trabalho"],
    "direito processual criminal": ["Direito Processual Penal"],
})


def explicit_areas(raw: str) -> list[str]:
    """Areas named verbatim by a component of the raw value."""
    areas: list[str] = []
    for part in re.split(r"\s*(?:/|;|,|–|—|\s-\s)\s*", raw):
        areas.extend(_EXPLICIT.get(normalize(part), []))
    return areas


def merge_areas(llm_areas: list[str], raw: str) -> list[str]:
    return list(dict.fromkeys([*llm_areas, *explicit_areas(raw)]))
BATCH = 100
CONCURRENCY = 4

PROMPT = """És um jurista português. Classifica cada "domínio jurídico" abaixo (texto livre
gerado automaticamente para acórdãos portugueses) numa ou mais áreas canónicas.

Áreas canónicas (usa exatamente estes nomes):
{areas}

Regras:
- Atribui todas as áreas que o texto menciona; normalmente 1 a 3.
- Sinónimos: "Fiscal"/"Aduaneiro" → Direito Tributário; "Criminal" → Direito Penal;
  "Laboral" → Direito do Trabalho; "Contencioso Administrativo" e "Expropriações" →
  Direito Administrativo; "Obrigações", "Contratos", "Reais", "Responsabilidade Civil" →
  Direito Civil; "Menores" → Direito da Família e Menores.
- "Processual Civil"/"Processo Civil" → Direito Processual Civil; "Processual Penal" →
  Direito Processual Penal.
- Usa lista vazia só se nenhuma área se aplicar.

Domínios a classificar (devolve um objeto por cada, com o texto "raw" exatamente igual):
{values}"""


def _schema() -> dict:
    return {
        "name": "legal_domain_mapping",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "raw": {"type": "string"},
                            "areas": {"type": "array", "items": {"type": "string", "enum": CANONICAL_AREAS}},
                        },
                        "required": ["raw", "areas"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["mappings"],
            "additionalProperties": False,
        },
    }


async def _map_batch(client: httpx.AsyncClient, values: list[str], sem: asyncio.Semaphore) -> dict[str, list[str]]:
    prompt = PROMPT.format(
        areas="\n".join(f"- {a}" for a in CANONICAL_AREAS),
        values="\n".join(json.dumps(v, ensure_ascii=False) for v in values),
    )
    async with sem:
        for attempt in range(4):
            try:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "response_format": {"type": "json_schema", "json_schema": _schema()},
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
                wanted = set(values)
                return {
                    m["raw"]: merge_areas([a for a in m["areas"] if a in CANONICAL_AREAS], m["raw"])
                    for m in parsed["mappings"]
                    if m["raw"] in wanted
                }
            except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
                if attempt == 3:
                    print(f"  batch failed: {exc}", flush=True)
                    return {}
                await asyncio.sleep(2 ** attempt)
    return {}


async def revalidate(conn: asyncpg.Connection) -> None:
    """Re-apply the explicit-name rule to existing mappings (no LLM calls)."""
    rows = await conn.fetch("SELECT raw, areas FROM legal_domain_map")
    changed = [(raw, merged) for raw, areas in rows
               if (merged := merge_areas(list(areas), raw)) != list(areas)]
    await conn.executemany("UPDATE legal_domain_map SET areas = $2 WHERE raw = $1", changed)
    print(f"Revalidated {len(rows)} mappings; {len(changed)} gained explicitly named areas")


async def run(dry_run: bool, revalidate_only: bool) -> None:
    conn = await asyncpg.connect(
        host=os.getenv("SUPABASE_DB_HOST"),
        port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
        user=os.getenv("SUPABASE_DB_USER"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        database=os.getenv("SUPABASE_DB_NAME", "postgres"),
        statement_cache_size=0,
        command_timeout=600,
    )
    if revalidate_only:
        await revalidate(conn)
        await conn.close()
        return
    pending = [r["legal_domain"] for r in await conn.fetch(
        "SELECT DISTINCT d.legal_domain FROM documents d "
        "WHERE d.legal_domain IS NOT NULL AND btrim(d.legal_domain) <> '' "
        "AND NOT EXISTS (SELECT 1 FROM legal_domain_map m WHERE m.raw = d.legal_domain)"
    )]
    print(f"{len(pending)} raw legal_domain values to map", flush=True)
    if dry_run or not pending:
        await conn.close()
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        # Two passes: values a batch answer omitted get retried in smaller batches.
        for batch_size in (BATCH, 20):
            pending_now = [v for v in pending if v not in await _mapped(conn, pending)]
            if not pending_now:
                break
            batches = [pending_now[i:i + batch_size] for i in range(0, len(pending_now), batch_size)]
            results = await asyncio.gather(*(_map_batch(client, b, sem) for b in batches))
            rows = [(raw, areas) for result in results for raw, areas in result.items()]
            await conn.executemany(
                "INSERT INTO legal_domain_map (raw, areas) VALUES ($1, $2) "
                "ON CONFLICT (raw) DO UPDATE SET areas = EXCLUDED.areas",
                rows,
            )
            print(f"  batch size {batch_size}: mapped {len(rows)} values", flush=True)

    unmapped = len(pending) - len(await _mapped(conn, pending))
    empty = await conn.fetchval("SELECT count(*) FROM legal_domain_map WHERE cardinality(areas) = 0")
    print(f"Done. Unmapped values left: {unmapped} | mapped to no area: {empty}")
    await conn.close()


async def _mapped(conn: asyncpg.Connection, values: list[str]) -> set[str]:
    rows = await conn.fetch("SELECT raw FROM legal_domain_map WHERE raw = ANY($1::text[])", values)
    return {r["raw"] for r in rows}


@click.command()
@click.option("--dry-run", is_flag=True)
@click.option("--revalidate", "revalidate_only", is_flag=True,
              help="Only re-apply the explicit-name rule to existing mappings")
def main(dry_run: bool, revalidate_only: bool) -> None:
    asyncio.run(run(dry_run, revalidate_only))


if __name__ == "__main__":
    main()
