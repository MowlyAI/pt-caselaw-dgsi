#!/usr/bin/env python3
"""Count AUJs per DB using fast string search."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scraper.config import DATABASES

ENH = Path("data/enhanced")
NEEDLE = b'"is_jurisprudence_unification": true'

total_auj = 0
for d in DATABASES:
    short = d["short"]
    enh_dir = ENH / short
    if not enh_dir.exists():
        print(f"{short}: NO DATA")
        continue
    count = 0
    lines = 0
    for f in sorted(enh_dir.glob("chunk_*.jsonl")):
        with open(f, "rb") as fh:
            for line in fh:
                lines += 1
                if NEEDLE in line:
                    count += 1
    total_auj += count
    print(f"{short}: {count} AUJs / {lines:,} lines")
print(f"TOTAL: {total_auj} AUJs")
