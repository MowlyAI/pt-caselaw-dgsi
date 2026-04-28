"""
Run the extractor on N docs across one or more courts, save each extraction to
disk, then compute a quality scorecard: per-field fill rate, avg list lengths,
language check, date-format check, enum validity on citation_context, etc.

Usage:
    python scripts/eval_extractor.py --dbs STJ,TRP,TRL,TCAS --n 20
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import click
import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv(".env.local")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extractor.extractor import extract_document

console = Console()
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/eval")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CITATION_CTX = {"supporting", "distinguishing", "criticizing", "referencing"}


def load_docs(dbs: list[str], n_per: int) -> list[dict]:
    out: list[dict] = []
    for short in dbs:
        db_dir = RAW_DIR / short
        if not db_dir.exists():
            continue
        files = sorted(db_dir.glob("chunk_*.jsonl"))
        taken = 0
        for path in files:
            if taken >= n_per:
                break
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if taken >= n_per:
                        break
                    if not line.strip():
                        continue
                    try:
                        out.append(json.loads(line))
                        taken += 1
                    except json.JSONDecodeError:
                        pass
    return out


def score(extractions: list[dict]) -> dict:
    n = len([e for e in extractions if e])
    s = {
        "total_attempted": len(extractions),
        "total_ok": n,
        "fill_rate": {},
        "list_avg_len": {},
        "dates_invalid": 0,
        "bad_citation_context": 0,
        "non_portuguese_query": 0,
        "confidence": Counter(),
    }
    if n == 0:
        return s
    scalar_fields = [
        "process_number", "court_name", "judge_name", "decision_date",
        "decision_type", "case_type", "summary", "legal_question",
        "decision_outcome", "ratio_decidendi", "legal_domain",
        "procedural_type", "instance_level", "semantic_search_query",
        "keywords_search_query",
    ]
    list_fields = [
        "legal_descriptors", "legislation_cited", "jurisprudence_cited",
        "doctrine_cited", "parties", "amounts_involved", "timeline_events",
        "insurance_companies", "injuries",
    ]
    bool_fields = ["documentary_evidence", "expert_testimony", "medical_evidence",
                   "witness_testimony", "liability_found"]
    filled = defaultdict(int)
    list_lens = defaultdict(list)
    for e in extractions:
        if not e:
            continue
        for f in scalar_fields:
            if e.get(f):
                filled[f] += 1
        for f in list_fields:
            v = e.get(f) or []
            if v:
                filled[f] += 1
            list_lens[f].append(len(v))
        for f in bool_fields:
            if e.get(f) is not None:
                filled[f] += 1
        d = e.get("decision_date")
        if d and not _DATE_RE.match(d):
            s["dates_invalid"] += 1
        for item in e.get("legislation_cited", []) or []:
            if (item or {}).get("citation_context") not in _CITATION_CTX:
                s["bad_citation_context"] += 1
        for item in e.get("jurisprudence_cited", []) or []:
            if (item or {}).get("citation_context") not in _CITATION_CTX:
                s["bad_citation_context"] += 1
        for item in e.get("doctrine_cited", []) or []:
            if (item or {}).get("citation_context") not in _CITATION_CTX:
                s["bad_citation_context"] += 1
        q = e.get("semantic_search_query") or ""
        # Heuristic: Portuguese queries should contain at least one of these markers
        if q and not re.search(r"[ãõçáéíóú]|\b(de|da|do|em|que|para|pela|pelo)\b", q, re.IGNORECASE):
            s["non_portuguese_query"] += 1
        s["confidence"][e.get("extraction_confidence") or "unset"] += 1
    s["fill_rate"] = {f: f"{filled[f]/n*100:4.0f}% ({filled[f]}/{n})" for f in scalar_fields + list_fields + bool_fields}
    s["list_avg_len"] = {f: f"{sum(list_lens[f])/n:.1f}" for f in list_fields}
    return s


@click.command()
@click.option("--dbs", default="STJ", help="comma-separated short codes")
@click.option("--n", default=20, type=int, help="total sample size")
@click.option("--concurrency", default=10, type=int)
def main(dbs: str, n: int, concurrency: int):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY missing[/red]"); sys.exit(1)
    db_list = [d.strip().upper() for d in dbs.split(",") if d.strip()]
    per_db = max(1, n // len(db_list))
    docs = load_docs(db_list, per_db)[:n]
    console.print(f"[cyan]Loaded {len(docs)} docs from {db_list} "
                  f"(avg {sum(len(d.get('full_text','')) for d in docs)//max(1,len(docs))} chars)[/cyan]")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    async def run():
        sem = asyncio.Semaphore(concurrency)
        usage: list = []
        async with httpx.AsyncClient() as client:
            t0 = time.perf_counter()
            results = await asyncio.gather(*[
                extract_document(client, d, api_key, sem, usage_sink=usage) for d in docs
            ], return_exceptions=True)
            return results, time.perf_counter() - t0, usage

    results, elapsed, usage = asyncio.run(run())
    clean: list[dict | None] = []
    for doc, res in zip(docs, results):
        if isinstance(res, Exception) or res is None:
            clean.append(None); continue
        clean.append(res)
        out = OUT_DIR / f"{doc['court_short']}_{doc['doc_id']}.json"
        out.write_text(json.dumps({"raw": doc, "extracted": res}, ensure_ascii=False, indent=2))
    s = score(clean)
    t = Table(title=f"Quality scorecard ({s['total_ok']}/{s['total_attempted']} ok, {elapsed:.1f}s)")
    t.add_column("metric"); t.add_column("value")
    for k, v in s["fill_rate"].items():
        t.add_row(f"fill · {k}", v)
    for k, v in s["list_avg_len"].items():
        t.add_row(f"avg_len · {k}", v)
    t.add_row("dates_invalid", str(s["dates_invalid"]))
    t.add_row("bad_citation_context", str(s["bad_citation_context"]))
    t.add_row("non_portuguese_query", str(s["non_portuguese_query"]))
    t.add_row("confidence", ", ".join(f"{k}={v}" for k, v in s["confidence"].items()))
    console.print(t)
    total_cost = sum(u.get("cost", 0.0) for u in usage)
    console.print(f"[yellow]total cost: ${total_cost:.4f}  "
                  f"(${total_cost/max(1,s['total_ok']):.4f}/doc, "
                  f"projected 387k: ${total_cost/max(1,s['total_ok'])*387_780:.0f})[/yellow]")
    console.print(f"[green]extractions saved → {OUT_DIR}/[/green]")


if __name__ == "__main__":
    main()
