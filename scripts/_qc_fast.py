#!/usr/bin/env python3
"""Fast data quality check using shell commands for line counting."""
import json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scraper.config import DATABASES

DATA_DIR = Path("data")
STATE_FILE = DATA_DIR / "extractor_state.json"

def wc_l(pattern):
    """Fast line count using wc -l."""
    try:
        r = subprocess.run(f"cat {pattern} 2>/dev/null | wc -l", shell=True, capture_output=True, text=True)
        return int(r.stdout.strip())
    except:
        return 0

def grep_count(pattern, field):
    """Count unique values of a field using grep + sort."""
    try:
        r = subprocess.run(
            f'grep -oE \'"{field}":"[^"]+"\' {pattern} 2>/dev/null | sort -u | wc -l',
            shell=True, capture_output=True, text=True)
        return int(r.stdout.strip())
    except:
        return 0

def grep_dup_count(pattern, field):
    """Count duplicate values."""
    try:
        r = subprocess.run(
            f'grep -oE \'"{field}":"[^"]+"\' {pattern} 2>/dev/null | sort | uniq -c | awk \'$1>1\' | wc -l',
            shell=True, capture_output=True, text=True)
        return int(r.stdout.strip())
    except:
        return 0

print("=" * 90)
print("EXTRACTION DATA QUALITY REPORT")
print("=" * 90)

# 1. State
state = json.load(open(STATE_FILE))
state_counts = {k: len(v) for k, v in state.get("processed_doc_ids", {}).items()}
state_total = sum(state_counts.values())
print(f"\n## 1. STATE FILE ({os.path.getsize(STATE_FILE)/1e6:.1f} MB) — TOTAL: {state_total:,}")
for k, v in sorted(state_counts.items()):
    print(f"    {k}: {v:,}")

# 2. Raw vs Enhanced
print(f"\n## 2. RAW vs ENHANCED LINE COUNTS")
print(f"  {'DB':<6} {'Label':<38} {'Expect':>7} {'Raw':>8} {'Enhanced':>8} {'State':>8} {'Dups':>6} {'Uniq':>8}")
print("  " + "-" * 92)
gt_raw = gt_enh = gt_state = gt_exp = 0

for d in DATABASES:
    short, db_key, label, expected = d["short"], d["db"], d["label"], d["approx_count"]
    raw_pat = f"data/raw/{short}/chunk_*.jsonl"
    enh_pat = f"data/enhanced/{short}/chunk_*.jsonl"
    raw_n = wc_l(raw_pat)
    enh_n = wc_l(enh_pat)
    s_n = state_counts.get(db_key, 0)
    uniq = grep_count(enh_pat, "doc_id")
    dups = grep_dup_count(enh_pat, "doc_id")
    gt_raw += raw_n; gt_enh += enh_n; gt_state += s_n; gt_exp += expected
    flag = ""
    if dups > 0: flag = " ⚠️ DUPS"
    if enh_n == 0: flag = " ❌ EMPTY"
    print(f"  {short:<6} {label:<38} {expected:>7,} {raw_n:>8,} {enh_n:>8,} {s_n:>8,} {dups:>6,} {uniq:>8,}{flag}")

print("  " + "-" * 92)
print(f"  {'TOTAL':<6} {'':<38} {gt_exp:>7,} {gt_raw:>8,} {gt_enh:>8,} {gt_state:>8,}")
print(f"\n  Coverage: {gt_state/gt_exp*100:.1f}% of expected | {gt_state/gt_raw*100:.1f}% of raw scraped")

# 3. JSON parse errors (sample first 500 lines per DB)
print(f"\n## 3. JSON VALIDITY (first 500 lines per DB)")
for d in DATABASES:
    short = d["short"]
    enh_dir = DATA_DIR / "enhanced" / short
    if not enh_dir.exists(): continue
    errors = 0; checked = 0
    for f in sorted(enh_dir.glob("chunk_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if checked >= 500: break
                checked += 1
                try:
                    json.loads(line.strip())
                except:
                    errors += 1
        if checked >= 500: break
    status = "✅" if errors == 0 else f"❌ {errors} errors"
    print(f"  {short}: {status} ({checked} checked)")

# 4. Field completeness (sample 100 per DB)
print(f"\n## 4. FIELD COMPLETENESS (sample 100 per DB)")
FIELDS = ["doc_id","url","case_number","date","court","summary","decision",
          "subject_area","legislation_refs","case_refs","judges","is_auj"]
for d in DATABASES:
    short = d["short"]
    enh_dir = DATA_DIR / "enhanced" / short
    if not enh_dir.exists(): continue
    from collections import Counter
    nonempty = Counter(); sampled = 0
    for f in sorted(enh_dir.glob("chunk_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if sampled >= 100: break
                try:
                    doc = json.loads(line)
                    for fld in FIELDS:
                        v = doc.get(fld)
                        if v is not None and v != "" and v != [] and v != {}: nonempty[fld] += 1
                except: pass
                sampled += 1
        if sampled >= 100: break
    print(f"  {short} ({sampled}):", end="")
    for fld in FIELDS:
        pct = nonempty.get(fld, 0) / sampled * 100 if sampled else 0
        marker = "✓" if pct >= 80 else ("~" if pct >= 50 else "✗")
        print(f" {fld}={marker}{pct:.0f}%", end="")
    print()

# 5. AUJ summary (use grep for speed)
print(f"\n## 5. AUJ DETECTION")
auj_total = 0
for d in DATABASES:
    short = d["short"]
    pat = f"data/enhanced/{short}/chunk_*.jsonl"
    try:
        r = subprocess.run(f'grep -c \'"is_auj": true\' {pat} 2>/dev/null || echo 0',
                          shell=True, capture_output=True, text=True)
        # Sum across files
        counts = [int(x) for x in r.stdout.strip().split('\n') if x.strip().isdigit()]
        auj = sum(counts)
    except:
        auj = 0
    total_lines = wc_l(pat)
    auj_total += auj
    pct = auj / total_lines * 100 if total_lines else 0
    print(f"  {short}: {auj:,} AUJs / {total_lines:,} lines ({pct:.2f}%)")
print(f"  TOTAL AUJs: {auj_total:,}")

# 6. Disk usage
print(f"\n## 6. DISK USAGE")
for d in DATABASES:
    short = d["short"]
    r = subprocess.run(f"du -sh data/raw/{short}/ data/enhanced/{short}/ 2>/dev/null",
                      shell=True, capture_output=True, text=True)
    lines = r.stdout.strip().split('\n')
    print(f"  {short}: " + " | ".join(l.strip() for l in lines if l.strip()))

r = subprocess.run("du -sh data/raw/ data/enhanced/", shell=True, capture_output=True, text=True)
print(f"\n  TOTALS: " + " | ".join(l.strip() for l in r.stdout.strip().split('\n')))

# 7. Date range check (sample 200 per DB)
print(f"\n## 7. DATE RANGE CHECK (sample 200 per DB)")
from datetime import datetime
for d in DATABASES:
    short = d["short"]
    enh_dir = DATA_DIR / "enhanced" / short
    if not enh_dir.exists(): continue
    dates = []; sampled = 0
    for f in sorted(enh_dir.glob("chunk_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if sampled >= 200: break
                sampled += 1
                try:
                    doc = json.loads(line)
                    dt = doc.get("date")
                    if dt:
                        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                            try: dates.append(datetime.strptime(str(dt), fmt)); break
                            except: pass
                except: pass
        if sampled >= 200: break
    if dates:
        print(f"  {short}: {min(dates).strftime('%Y-%m-%d')} → {max(dates).strftime('%Y-%m-%d')} ({len(dates)}/{sampled} valid)")
    else:
        print(f"  {short}: no valid dates")

print("\n" + "=" * 90)
print(f"TOTAL EXTRACTED: {gt_state:,} unique docs | COVERAGE: {gt_state/gt_exp*100:.1f}%")
print("=" * 90)
