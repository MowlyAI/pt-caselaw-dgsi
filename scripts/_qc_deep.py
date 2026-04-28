#!/usr/bin/env python3
"""Deep quality check — correct field paths under llm_extracted."""
import json, sys, subprocess
from pathlib import Path
from collections import Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from scraper.config import DATABASES

ENH = Path("data/enhanced")

def wc_l(pat):
    r = subprocess.run(f"wc -l {pat} 2>/dev/null | tail -1", shell=True, capture_output=True, text=True)
    try: return int(r.stdout.strip().split()[0])
    except: return 0

# 1. Duplicate doc_id analysis per DB (Python-based, full scan)
print("## 1. DUPLICATE DOC_ID ANALYSIS")
for d in DATABASES:
    short = d["short"]
    enh_dir = ENH / short
    if not enh_dir.exists():
        print(f"  {short}: NO DATA"); continue
    seen = Counter()
    total = 0
    for f in sorted(enh_dir.glob("chunk_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                total += 1
                try:
                    # Fast: just extract doc_id via string search
                    idx = line.find('"doc_id":"')
                    if idx >= 0:
                        start = idx + 10
                        end = line.index('"', start)
                        seen[line[start:end]] += 1
                except: pass
    dups = sum(1 for v in seen.values() if v > 1)
    dup_lines = sum(v - 1 for v in seen.values() if v > 1)
    uniq = len(seen)
    if dups > 0:
        print(f"  {short}: {total:,} lines, {uniq:,} unique — ⚠️  {dups:,} IDs duplicated ({dup_lines:,} extra lines)")
    else:
        print(f"  {short}: {total:,} lines, {uniq:,} unique — ✅ no duplicates")

# 2. Field completeness (nested under llm_extracted)
print("\n## 2. FIELD COMPLETENESS (sample 200 per DB)")
TOP_FIELDS = ["doc_id", "url", "source_db", "court", "full_text"]
LLM_FIELDS = ["process_number", "court_name", "judge_name", "decision_date",
              "decision_type", "case_type", "summary", "legal_question",
              "decision_outcome", "ratio_decidendi", "legal_descriptors",
              "legislation_cited", "is_auj", "auj_number"]

for d in DATABASES:
    short = d["short"]
    enh_dir = ENH / short
    if not enh_dir.exists(): continue
    top_cnt = Counter(); llm_cnt = Counter(); sampled = 0
    for f in sorted(enh_dir.glob("chunk_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if sampled >= 200: break
                try:
                    doc = json.loads(line)
                    for fld in TOP_FIELDS:
                        v = doc.get(fld)
                        if v is not None and v != "" and v != []: top_cnt[fld] += 1
                    llm = doc.get("llm_extracted", {})
                    for fld in LLM_FIELDS:
                        v = llm.get(fld)
                        if v is not None and v != "" and v != [] and v != {}: llm_cnt[fld] += 1
                except: pass
                sampled += 1
        if sampled >= 200: break
    print(f"\n  {short} ({sampled} docs):")
    print(f"    Top-level:", end="")
    for fld in TOP_FIELDS:
        pct = top_cnt.get(fld, 0) / sampled * 100
        print(f" {fld}={'✓' if pct>=80 else '✗'}{pct:.0f}%", end="")
    print(f"\n    LLM fields:", end="")
    for fld in LLM_FIELDS:
        pct = llm_cnt.get(fld, 0) / sampled * 100
        print(f" {fld}={'✓' if pct>=80 else ('~' if pct>=50 else '✗')}{pct:.0f}%", end="")
    print()

# 3. AUJ detection (nested)
print("\n## 3. AUJ DETECTION (full scan)")
auj_total = 0
for d in DATABASES:
    short = d["short"]
    pat = f"data/enhanced/{short}/chunk_*.jsonl"
    total = wc_l(pat)
    # grep for is_auj with any whitespace variation
    r = subprocess.run(f'grep -cE \'"is_auj"\\s*:\\s*true\' {pat} 2>/dev/null || echo 0',
                      shell=True, capture_output=True, text=True)
    counts = [int(x) for x in r.stdout.strip().split('\n') if x.strip().isdigit()]
    auj = sum(counts)
    auj_total += auj
    pct = auj / total * 100 if total else 0
    print(f"  {short}: {auj:,} AUJs / {total:,} ({pct:.2f}%)")
print(f"  TOTAL: {auj_total:,}")

# 4. Date range (nested under llm_extracted.decision_date)
print("\n## 4. DATE RANGE (sample 300 per DB)")
for d in DATABASES:
    short = d["short"]
    enh_dir = ENH / short
    if not enh_dir.exists(): continue
    dates = []; no_date = 0; sampled = 0
    for f in sorted(enh_dir.glob("chunk_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if sampled >= 300: break
                sampled += 1
                try:
                    doc = json.loads(line)
                    dt = doc.get("llm_extracted", {}).get("decision_date")
                    if not dt: no_date += 1; continue
                    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                        try: dates.append(datetime.strptime(str(dt), fmt)); break
                        except: pass
                except: pass
        if sampled >= 300: break
    if dates:
        mn, mx = min(dates), max(dates)
        future = sum(1 for d in dates if d.year > 2026)
        print(f"  {short}: {mn.strftime('%Y-%m-%d')} → {mx.strftime('%Y-%m-%d')} ({len(dates)}/{sampled} valid, {future} future)")
    else:
        print(f"  {short}: no valid dates (no_date={no_date})")

# 5. Extraction failures summary (from log)
print("\n## 5. EXTRACTION LOG SUMMARY")
r = subprocess.run("grep -c 'fail=' data/logs/extract_full.log 2>/dev/null || echo 0",
                  shell=True, capture_output=True, text=True)
r2 = subprocess.run("grep '✓ Extracted' data/logs/extract_full.log 2>/dev/null",
                   shell=True, capture_output=True, text=True)
for line in r2.stdout.strip().split('\n'):
    if line.strip(): print(f"  {line.strip()}")

print("\n## 6. SAMPLE LLM OUTPUT QUALITY (1 random doc per DB)")
import random
for d in DATABASES:
    short = d["short"]
    enh_dir = ENH / short
    if not enh_dir.exists(): continue
    chunks = sorted(enh_dir.glob("chunk_*.jsonl"))
    if not chunks: continue
    with open(chunks[0]) as fh:
        lines = [fh.readline() for _ in range(50)]
    line = random.choice([l for l in lines if l.strip()])
    doc = json.loads(line)
    llm = doc.get("llm_extracted", {})
    summary = llm.get("summary", "N/A")[:120]
    print(f"  {short}: {llm.get('process_number','?')} | {llm.get('decision_date','?')} | {llm.get('decision_type','?')} | {summary}...")
