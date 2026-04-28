"""
Benchmark multiple OpenRouter models on the same 20 legal docs.

For each model, runs the exact same extraction pipeline used in production
(same SYSTEM_PROMPT, same JSON schema, same parser), records per-model:
  - success rate
  - wall-clock time
  - prompt / completion / cached tokens
  - reported cost
  - quality scorecard (fill rates, list lengths, format validity)

Reuses the 20 raw docs already in data/eval/*.json (produced by eval_extractor.py).
"""
from __future__ import annotations

import asyncio
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
from extractor import extractor as extractor_mod
from extractor.extractor import extract_document

console = Console()
EVAL_DIR = Path("data/eval")
BENCH_DIR = Path("data/eval/benchmark")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CITATION_CTX = {"supporting", "distinguishing", "criticizing", "referencing"}

MODELS = [
    "xiaomi/mimo-v2-flash",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-2.5-flash-lite",
    "openai/gpt-4.1-nano",
    "qwen/qwen3.5-flash-02-23",
    "openai/gpt-5-nano",
]


def load_eval_docs() -> list[dict]:
    out = []
    for f in sorted(EVAL_DIR.glob("*.json")):
        if f.parent == EVAL_DIR:
            try:
                d = json.loads(f.read_text())
                if "raw" in d:
                    out.append(d["raw"])
            except Exception:
                pass
    return out


def score(extractions: list[dict | None]) -> dict:
    ok = [e for e in extractions if isinstance(e, dict)]
    n = len(ok)
    s: dict = {
        "ok": n,
        "total": len(extractions),
        "fill_rate": {},
        "list_avg": {},
        "dates_invalid": 0,
        "bad_citation_ctx": 0,
        "non_pt_query": 0,
        "confidence": Counter(),
    }
    scalars = ["process_number", "court_name", "judge_name", "decision_date",
               "decision_type", "summary", "legal_question", "ratio_decidendi",
               "legal_domain", "procedural_type", "semantic_search_query",
               "keywords_search_query"]
    lists = ["legal_descriptors", "legislation_cited", "jurisprudence_cited",
             "doctrine_cited", "parties", "amounts_involved", "timeline_events"]
    filled = defaultdict(int); lens = defaultdict(list)
    for e in ok:
        for f in scalars:
            if e.get(f): filled[f] += 1
        for f in lists:
            v = e.get(f) or []
            if v: filled[f] += 1
            lens[f].append(len(v))
        if e.get("decision_date") and not _DATE_RE.match(e["decision_date"]):
            s["dates_invalid"] += 1
        for key in ("legislation_cited", "jurisprudence_cited", "doctrine_cited"):
            for item in e.get(key) or []:
                if (item or {}).get("citation_context") not in _CITATION_CTX:
                    s["bad_citation_ctx"] += 1
        q = e.get("semantic_search_query") or ""
        if q and not re.search(r"[ãõçáéíóú]|\b(de|da|do|em|que|para|pela|pelo)\b", q, re.I):
            s["non_pt_query"] += 1
        s["confidence"][e.get("extraction_confidence") or "unset"] += 1
    s["fill_rate"] = {f: filled[f] / max(1, n) for f in scalars + lists}
    s["list_avg"] = {f: sum(lens[f]) / max(1, n) for f in lists}
    return s


async def run_model(model: str, docs: list[dict], api_key: str, concurrency: int) -> dict:
    extractor_mod.MODEL = model
    sem = asyncio.Semaphore(concurrency)
    usage: list = []
    out_dir = BENCH_DIR / model.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            extract_document(client, d, api_key, sem, usage_sink=usage) for d in docs
        ], return_exceptions=True)
        elapsed = time.perf_counter() - t0
    clean: list[dict | None] = []
    for doc, res in zip(docs, results):
        if isinstance(res, Exception) or res is None:
            clean.append(None); continue
        clean.append(res)
        (out_dir / f"{doc['court_short']}_{doc['doc_id']}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2)
        )
    prompt_tokens = sum(u.get("prompt_tokens", 0) for u in usage)
    completion_tokens = sum(u.get("completion_tokens", 0) for u in usage)
    cached = 0
    for u in usage:
        ptd = u.get("prompt_tokens_details") or {}
        cached += ptd.get("cached_tokens", 0) or u.get("cache_read_input_tokens", 0) or 0
    cost = sum(u.get("cost", 0.0) for u in usage)
    return {
        "model": model, "elapsed_s": elapsed, "scorecard": score(clean),
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "cached": cached, "cost_usd": cost,
        "extractions": clean,
    }


@click.command()
@click.option("--concurrency", default=10, type=int)
@click.option("--models", default=",".join(MODELS), help="comma-separated overrides")
def main(concurrency: int, models: str):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY missing[/red]"); sys.exit(1)
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    docs = load_eval_docs()
    if not docs:
        console.print("[red]No raw docs in data/eval/. Run eval_extractor.py first.[/red]"); sys.exit(1)
    console.print(f"[cyan]{len(docs)} docs · {len(model_list)} models · concurrency {concurrency}[/cyan]")

    all_results = []
    for m in model_list:
        console.print(f"[bold yellow]→ {m}[/bold yellow]")
        try:
            r = asyncio.run(run_model(m, docs, api_key, concurrency))
        except Exception as e:
            console.print(f"[red]{m} FAILED: {type(e).__name__}: {e}[/red]")
            continue
        all_results.append(r)
        sc = r["scorecard"]
        console.print(f"  ok={sc['ok']}/{sc['total']}  {r['elapsed_s']:.1f}s  "
                      f"${r['cost_usd']:.4f}  dates_bad={sc['dates_invalid']}  "
                      f"enum_bad={sc['bad_citation_ctx']}  non_pt={sc['non_pt_query']}")
    (BENCH_DIR / "summary.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "extractions"} for r in all_results],
        ensure_ascii=False, indent=2, default=str
    ))
    _print_compare_table(all_results)


def _print_compare_table(results: list[dict]):
    t = Table(title="Model benchmark (same 20 docs)", expand=True)
    t.add_column("model"); t.add_column("ok", justify="right"); t.add_column("s", justify="right")
    t.add_column("cost", justify="right"); t.add_column("prj 387k", justify="right")
    t.add_column("p.tok", justify="right"); t.add_column("c.tok", justify="right")
    t.add_column("bad_dt", justify="right"); t.add_column("bad_enum", justify="right"); t.add_column("non_pt", justify="right")
    t.add_column("fill core"); t.add_column("avg leg/jur/doct/parties")
    core_fields = ["process_number", "summary", "ratio_decidendi", "legal_question", "semantic_search_query"]
    for r in sorted(results, key=lambda x: x["cost_usd"]):
        sc = r["scorecard"]
        proj = r["cost_usd"] / max(1, sc["ok"]) * 387_780
        core_fill = sum(sc["fill_rate"].get(f, 0) for f in core_fields) / len(core_fields) * 100
        avgs = f"{sc['list_avg'].get('legislation_cited', 0):.1f}/{sc['list_avg'].get('jurisprudence_cited', 0):.1f}/{sc['list_avg'].get('doctrine_cited', 0):.1f}/{sc['list_avg'].get('parties', 0):.1f}"
        t.add_row(
            r["model"].split("/")[-1][:30], f"{sc['ok']}/{sc['total']}", f"{r['elapsed_s']:.0f}",
            f"${r['cost_usd']:.4f}", f"${proj:.0f}",
            f"{r['prompt_tokens']:,}", f"{r['completion_tokens']:,}",
            str(sc["dates_invalid"]), str(sc["bad_citation_ctx"]), str(sc["non_pt_query"]),
            f"{core_fill:.0f}%", avgs,
        )
    console.print(t)


if __name__ == "__main__":
    main()
