"""
Live monitor for the DGSI scraper.

Reads data/scraper_state.json + data/raw/ every few seconds and prints a
rich table with per-DB progress, throughput (docs/sec over a rolling window)
and a total-ETA projection.

Usage:
    python scripts/monitor.py              # refresh every 3 s
    python scripts/monitor.py --interval 1
    python scripts/monitor.py --once       # print once and exit (good for cron/tail)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scraper.config import DATABASES

STATE = Path("data/scraper_state.json")
RAW = Path("data/raw")
LOG = Path("data/logs/scraper.log")


def scraper_pid() -> int | None:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "scraper.runner"], text=True
        ).strip().splitlines()
        for pid in out:
            p = pid.strip()
            if p.isdigit() and int(p) != os.getpid():
                return int(p)
    except subprocess.CalledProcessError:
        return None
    return None


def read_state() -> dict:
    if not STATE.exists():
        return {"scraped_doc_ids": {}, "completed_dbs": []}
    try:
        return json.loads(STATE.read_text())
    except json.JSONDecodeError:
        return {"scraped_doc_ids": {}, "completed_dbs": []}


def bytes_of(short: str) -> int:
    p = RAW / short
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.glob("*.jsonl"))


def human_mb(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    return f"{n / (1 << 20):.1f} MB"


def human_dur(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def build_table(state: dict, prev_total: int, prev_time: float,
                window: deque) -> tuple[Table, int, Panel]:
    scraped = state.get("scraped_doc_ids", {})
    completed = set(state.get("completed_dbs", []))
    total_scraped = sum(len(v) for v in scraped.values())
    total_target = sum(d["approx_count"] for d in DATABASES)

    now = time.time()
    if prev_time:
        dt = max(now - prev_time, 1e-6)
        window.append(((total_scraped - prev_total) / dt, dt))
    inst_rate = sum(r * w for r, w in window) / max(sum(w for _, w in window), 1e-6) if window else 0.0

    eta = (total_target - total_scraped) / inst_rate if inst_rate > 0 else None

    table = Table(title="DGSI Scraper — live progress", expand=True)
    table.add_column("Court", style="bold cyan")
    table.add_column("Scraped", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Disk", justify="right")
    table.add_column("Status", justify="left")

    for d in DATABASES:
        n = len(scraped.get(d["db"], []))
        tgt = d["approx_count"]
        pct = (n / tgt * 100) if tgt else 0.0
        status = "[green]done[/green]" if d["db"] in completed else (
            "[yellow]running[/yellow]" if n > 0 else "[dim]pending[/dim]"
        )
        table.add_row(
            f'{d["short"]}  {d["label"][:28]}',
            f"{n:,}",
            f"{tgt:,}",
            f"{pct:5.1f}",
            human_mb(bytes_of(d["short"])),
            status,
        )

    pid = scraper_pid()
    alive = Text(f"PID {pid} alive", style="green") if pid else Text("NOT RUNNING", style="red bold")
    header_text = Text.assemble(
        alive,
        ("   total  ", ""),
        (f"{total_scraped:>7,} / {total_target:,}   ", "bold"),
        (f"{total_scraped / total_target * 100:5.1f}%   ", "bold"),
        ("rate  ", ""),
        (f"{inst_rate:6.1f} docs/s   ", "cyan"),
        ("ETA  ", ""),
        (human_dur(eta) if eta else "—", "bold yellow"),
    )
    header = Panel.fit(header_text, border_style="blue")
    return table, total_scraped, header


@click.command()
@click.option("--interval", default=3.0, type=float, help="refresh seconds")
@click.option("--once", is_flag=True, help="print once and exit")
def main(interval: float, once: bool):
    console = Console()
    window: deque = deque(maxlen=30)  # rolling throughput window
    prev_total, prev_time = 0, 0.0

    def render():
        nonlocal prev_total, prev_time
        state = read_state()
        table, new_total, header = build_table(state, prev_total, prev_time, window)
        prev_total, prev_time = new_total, time.time()
        grid = Table.grid()
        grid.add_row(header)
        grid.add_row(table)
        return grid

    if once:
        console.print(render())
        return

    try:
        with Live(render(), refresh_per_second=4, console=console, screen=False) as live:
            while True:
                time.sleep(interval)
                live.update(render())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
