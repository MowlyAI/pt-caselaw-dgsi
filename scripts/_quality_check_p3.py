#!/usr/bin/env python3
"""Part 3: Schema validation, date sanity, JSON parse errors, disk usage."""
import json
import os
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from scraper.config import DATABASES

ENH_DIR = Path("data/enhanced")

print("\n## 6. JSON PARSE ERRORS & MALFORMED LINES")
for db_info in DATABASES:
    short = db_info["short"]
    enh_dir = ENH_DIR / short
    if not enh_dir.exists():
        continue
    errors = 0
    empty_lines = 0
    total = 0
    for f in sorted(enh_dir.glob("chunk_*.jsonl")):
        with open(f) as fh:
            for i, line in enumerate(fh, 1):
                total += 1
                stripped = line.strip()
                if not stripped:
                    empty_lines += 1
                    continue
                try:
                    json.loads(stripped)
                except json.JSONDecodeError:
                    errors += 1
                    if errors <= 3:
                        print(f"    {short}/{f.name}:{i} — parse error: {stripped[:80]}...")
    if errors > 0 or empty_lines > 0:
        print(f"  {short}: {errors} parse errors, {empty_lines} empty lines / {total:,} total")
    else:
        print(f"  {short}: ✅ all {total:,} lines valid JSON")

print("\n## 7. DATE SANITY CHECK (sample all docs)")
date_errors = Counter()
date_range = {}  # db -> (min, max)
for db_info in DATABASES:
    short = db_info["short"]
    enh_dir = ENH_DIR / short
    if not enh_dir.exists():
        continue
    dates = []
    no_date = 0
    bad_date = 0
    total = 0
    for f in sorted(enh_dir.glob("chunk_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                total += 1
                try:
                    d = json.loads(line)
                    dt = d.get("date")
                    if not dt:
                        no_date += 1
                        continue
                    # Try parsing
                    parsed = None
                    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                        try:
                            parsed = datetime.strptime(str(dt), fmt)
                            break
                        except:
                            pass
                    if parsed:
                        dates.append(parsed)
                    else:
                        bad_date += 1
                except:
                    pass
    if dates:
        mn = min(dates)
        mx = max(dates)
        date_range[short] = (mn, mx)
        future = sum(1 for d in dates if d.year > 2026)
        ancient = sum(1 for d in dates if d.year < 1900)
        print(f"  {short}: range {mn.strftime('%Y-%m-%d')} → {mx.strftime('%Y-%m-%d')}, "
              f"no_date={no_date}, bad_format={bad_date}, future={future}, pre-1900={ancient}")
    else:
        print(f"  {short}: no valid dates found (no_date={no_date}, bad={bad_date})")

print("\n## 8. DISK USAGE")
for db_info in DATABASES:
    short = db_info["short"]
    raw_dir = Path("data/raw") / short
    enh_dir = ENH_DIR / short
    raw_size = sum(f.stat().st_size for f in raw_dir.glob("chunk_*.jsonl")) if raw_dir.exists() else 0
    enh_size = sum(f.stat().st_size for f in enh_dir.glob("chunk_*.jsonl")) if enh_dir.exists() else 0
    print(f"  {short}: raw={raw_size/1e6:.0f}MB  enhanced={enh_size/1e6:.0f}MB  ratio={enh_size/raw_size:.1f}x" if raw_size > 0 else f"  {short}: raw=0MB  enhanced={enh_size/1e6:.0f}MB")

raw_total = sum(f.stat().st_size for f in Path("data/raw").rglob("chunk_*.jsonl"))
enh_total = sum(f.stat().st_size for f in Path("data/enhanced").rglob("chunk_*.jsonl"))
print(f"\n  TOTAL: raw={raw_total/1e9:.2f}GB  enhanced={enh_total/1e9:.2f}GB")

print("\n## 9. COMPLETION STATUS")
state = json.load(open("data/extractor_state.json"))
state_counts = {k: len(v) for k, v in state.get("processed_doc_ids", {}).items()}
all_done = True
for db_info in DATABASES:
    short = db_info["short"]
    db_key = db_info["db"]
    raw_dir = Path("data/raw") / short
    raw_count = sum(1 for f in raw_dir.glob("chunk_*.jsonl") for _ in open(f)) if raw_dir.exists() else 0
    s_count = state_counts.get(db_key, 0)
    remaining = raw_count - s_count
    status = "✅ COMPLETE" if remaining <= 0 else f"❌ {remaining:,} remaining"
    if remaining > 0:
        all_done = False
    print(f"  {short}: {status} ({s_count:,}/{raw_count:,})")

print("\n" + "=" * 80)
if all_done:
    print("🎉 ALL DATABASES FULLY EXTRACTED")
else:
    print("⚠️  SOME DATABASES HAVE REMAINING DOCUMENTS")
print("=" * 80)
