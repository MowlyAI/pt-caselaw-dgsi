"""Run the full incremental DGSI refresh, in order.

    scrape → extract → embed+upload → legislation citations → legislation
    embeddings → citation doc_counts → canonical legal areas → duplicate
    marking → fts_core gap check → disability points

Every step only processes what is new, so this is safe to re-run. Pass
--from-step to resume after a failure (keep the same --since so the
incremental steps still cover the whole refresh).

Usage:
    python scripts/run_refresh.py
    python scripts/run_refresh.py --from-step embed --since 2026-10-08
    python scripts/run_refresh.py --list
"""
import shutil
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import click

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _steps(since: str) -> list[tuple[str, list[str]]]:
    return [
        ("scrape", [PY, "-m", "scraper.runner", "--refresh", "--concurrency", "40"]),
        ("extract", [PY, "-m", "extractor.runner", "--concurrency", "80"]),
        ("embed", [PY, "-m", "embedder.runner"]),
        ("citations", [PY, "scripts/create_legislation_citations_table.py", "--updated-since", since]),
        ("legislation-embeddings", [PY, "scripts/build_legislation_embeddings.py"]),
        ("doc-counts", [PY, "scripts/_backfill_doc_count.py"]),
        ("legal-areas", [PY, "scripts/canonicalize_legal_domains.py"]),
        ("duplicates", [PY, "scripts/mark_duplicates.py", "--updated-since", since]),
        ("fts-core", [PY, "scripts/backfill_fts_core.py"]),
        ("disability-points", [PY, "scripts/backfill_disability_points.py", "--updated-since", since]),
    ]


def _backup_state() -> None:
    backup = ROOT / "data" / f"state_backup_{datetime.now():%Y%m%d_%H%M}"
    backup.mkdir(parents=True, exist_ok=True)
    for name in ("scraper_state.json", "extractor_state.json", "embedder_state.json"):
        src = ROOT / "data" / name
        if src.exists():
            shutil.copy2(src, backup / name)
    print(f"State files backed up to {backup}", flush=True)


@click.command()
@click.option("--since", default=None,
              help="Refresh start date for incremental steps (default: today)")
@click.option("--from-step", default=None, help="Resume from this step")
@click.option("--list", "list_steps", is_flag=True, help="List steps and exit")
def main(since: Optional[str], from_step: Optional[str], list_steps: bool) -> None:
    since = since or date.today().isoformat()
    steps = _steps(since)
    names = [name for name, _ in steps]
    if list_steps:
        print("\n".join(names))
        return
    if from_step:
        if from_step not in names:
            raise click.BadParameter(f"unknown step; choose from {names}", param_hint="--from-step")
        steps = steps[names.index(from_step):]
    else:
        _backup_state()

    for name, cmd in steps:
        print(f"\n=== {name}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"\nStep '{name}' failed (exit {result.returncode}). Resume with:\n"
                  f"  python scripts/run_refresh.py --from-step {name} --since {since}")
            sys.exit(result.returncode)
        print(f"=== {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    print("\nRefresh complete.")


if __name__ == "__main__":
    main()
