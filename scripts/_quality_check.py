#!/usr/bin/env python3
"""Extensive data quality check for the extracted corpus."""
import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from scraper.config import DATABASES

# Build db map
db_map = {d['short']: d['db'] for d in DATABASES}
db_names = {d['short']: d['label'] for d in DATABASES}
db_expected = {d['short']: d['approx_count'] for d in DATABASES}

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
ENH_DIR = DATA_DIR / "enhanced"
STATE_FILE = DATA_DIR / "extractor_state.json"

print("=" * 80)
print("EXTRACTION DATA QUALITY REPORT")
print("=" * 80)

# 1. State file check
print("\n## 1. STATE FILE")
state = json.load(open(STATE_FILE))
state_counts = {k: len(v) for k, v in state.get("processed_doc_ids", {}).items()}
state_total = sum(state_counts.values())
print(f"  State file: {STATE_FILE} ({os.path.getsize(STATE_FILE)/1e6:.1f} MB)")
for db_key, count in sorted(state_counts.items()):
    print(f"    {db_key}: {count:,}")
print(f"  TOTAL in state: {state_total:,}")

# 2. Raw vs Enhanced counts per database
print("\n## 2. RAW vs ENHANCED COUNTS PER DATABASE")
print(f"  {'DB':<6} {'Name':<35} {'Expected':>8} {'Raw':>8} {'Enhanced':>8} {'State':>8} {'Δ Enh-State':>11}")
print("  " + "-" * 78)

grand_raw = 0
grand_enh = 0
grand_state = 0
grand_expected = 0

for db_info in DATABASES:
    short = db_info["short"]
    db_key = db_info["db"]
    name = db_info["label"]
    expected = db_info["approx_count"]

    # Count raw lines
    raw_dir = RAW_DIR / short
    raw_count = 0
    if raw_dir.exists():
        for f in sorted(raw_dir.glob("chunk_*.jsonl")):
            with open(f) as fh:
                for _ in fh:
                    raw_count += 1

    # Count enhanced lines
    enh_dir = ENH_DIR / short
    enh_count = 0
    if enh_dir.exists():
        for f in sorted(enh_dir.glob("chunk_*.jsonl")):
            with open(f) as fh:
                for _ in fh:
                    enh_count += 1

    s_count = state_counts.get(db_key, 0)
    delta = enh_count - s_count

    grand_raw += raw_count
    grand_enh += enh_count
    grand_state += s_count
    grand_expected += expected

    flag = ""
    if enh_count == 0:
        flag = " ❌ NO DATA"
    elif delta > raw_count * 0.05:
        flag = " ⚠️  DUPLICATES?"
    elif raw_count > 0 and enh_count < raw_count * 0.95:
        flag = " ⚠️  LOW YIELD"

    print(f"  {short:<6} {name:<35} {expected:>8,} {raw_count:>8,} {enh_count:>8,} {s_count:>8,} {delta:>+11,}{flag}")

print("  " + "-" * 78)
print(f"  {'TOTAL':<6} {'':<35} {grand_expected:>8,} {grand_raw:>8,} {grand_enh:>8,} {grand_state:>8,} {grand_enh - grand_state:>+11,}")
print(f"\n  Coverage: {grand_state/grand_expected*100:.1f}% of expected | {grand_state/grand_raw*100:.1f}% of raw")
