"""
Small test harness for the LLM extractor.

Picks a handful of docs from one or more raw shards, runs `extract_document`
against them in parallel, times the run, prints a compact summary + one
full JSON pretty-printed so quality can be inspected by eye.

Usage:
    python scripts/test_extractor.py                       # 5 docs from STJ
    python scripts/test_extractor.py --db TRP --n 10
    python scripts/test_extractor.py --model xiaomi/mimo-v2-flash --n 8
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import sys
import time
from pathlib import Path

import click
import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv(".env.local")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extractor import extractor as extractor_mod
from extractor.extractor import extract_document

console = Console()
RAW_DIR = Path("data/raw")


def iter_docs(db_short: str, limit: int):
    db_dir = RAW_DIR / db_short
    files = sorted(db_dir.glob("chunk_*.jsonl"))
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def sample(iterable, n):
    return list(itertools.islice(iterable, n))


@click.command()
@click.option("--db", default="STJ")
@click.option("--n", default=5, type=int)
@click.option("--concurrency", default=5, type=int)
@click.option("--model", default=None, help="override LLM_MODEL env")
@click.option("--show-full", is_flag=True, help="pretty-print every extraction")
def main(db: str, n: int, concurrency: int, model: str | None, show_full: bool):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY missing from .env.local[/red]")
        sys.exit(1)

    if model:
        extractor_mod.MODEL = model
    console.print(f"[cyan]Using model:[/cyan] {extractor_mod.MODEL}")

    docs = sample(iter_docs(db, n), n)
    if not docs:
        console.print(f"[red]No docs found for {db} in {RAW_DIR}[/red]")
        sys.exit(1)
    console.print(f"[cyan]Loaded {len(docs)} {db} docs[/cyan]  "
                  f"(avg len {sum(len(d.get('full_text','')) for d in docs)//len(docs)} chars)")

    async def run():
        sem = asyncio.Semaphore(concurrency)
        usage_sink: list = []
        async with httpx.AsyncClient() as client:
            t0 = time.perf_counter()
            results = await asyncio.gather(*[
                extract_document(client, d, api_key, sem, usage_sink=usage_sink)
                for d in docs
            ], return_exceptions=True)
            return results, time.perf_counter() - t0, usage_sink

    results, elapsed, usage_sink = asyncio.run(run())

    table = Table(title="extraction results", expand=True)
    table.add_column("#", justify="right")
    table.add_column("process_number")
    table.add_column("decision_date")
    table.add_column("legal_domain")
    table.add_column("confidence")
    table.add_column("#leg", justify="right")
    table.add_column("#jur", justify="right")
    table.add_column("#doct", justify="right")
    table.add_column("#parties", justify="right")
    table.add_column("#injuries", justify="right")
    table.add_column("status")

    ok = 0
    for i, (doc, res) in enumerate(zip(docs, results)):
        if isinstance(res, Exception) or res is None:
            table.add_row(str(i), "—", "—", "—", "—", "—", "—", "—", "—", "—",
                          f"[red]{type(res).__name__ if isinstance(res, Exception) else 'None'}[/red]")
            continue
        ok += 1
        table.add_row(
            str(i),
            (res.get("process_number") or "—")[:20],
            res.get("decision_date") or "—",
            (res.get("legal_domain") or "—")[:20],
            res.get("extraction_confidence") or "—",
            str(len(res.get("legislation_cited") or [])),
            str(len(res.get("jurisprudence_cited") or [])),
            str(len(res.get("doctrine_cited") or [])),
            str(len(res.get("parties") or [])),
            str(len(res.get("injuries") or [])),
            "[green]ok[/green]",
        )

    console.print(table)

    prompt_tokens = sum(u.get("prompt_tokens", 0) for u in usage_sink)
    cached = 0
    for u in usage_sink:
        ptd = u.get("prompt_tokens_details") or {}
        cached += ptd.get("cached_tokens", 0) or u.get("cache_read_input_tokens", 0) or 0
    completion = sum(u.get("completion_tokens", 0) for u in usage_sink)
    total_cost = sum(u.get("cost", 0.0) for u in usage_sink)
    cache_hit_pct = (cached / prompt_tokens * 100) if prompt_tokens else 0.0

    console.print(Panel.fit(
        f"success: [bold green]{ok}/{len(docs)}[/bold green]   "
        f"elapsed: [bold]{elapsed:.1f}s[/bold]   "
        f"avg: [bold]{elapsed/len(docs):.2f}s/doc[/bold]   "
        f"effective QPS: [bold]{len(docs)/elapsed:.1f}[/bold]\n"
        f"prompt tokens: [bold]{prompt_tokens:,}[/bold]   "
        f"cached: [bold cyan]{cached:,} ({cache_hit_pct:.1f}%)[/bold cyan]   "
        f"completion: [bold]{completion:,}[/bold]   "
        f"cost (OR-reported): [bold yellow]${total_cost:.4f}[/bold yellow]",
        border_style="blue",
    ))

    first_ok = next((r for r in results if isinstance(r, dict)), None)
    if first_ok:
        console.print(Panel.fit(
            json.dumps(first_ok, ensure_ascii=False, indent=2),
            title="first successful extraction",
            border_style="cyan",
        ))
    if show_full:
        for i, res in enumerate(results):
            if isinstance(res, dict):
                console.print(f"--- doc {i} ---")
                console.print_json(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
