#!/usr/bin/env python3
"""Quality check v2 — sampling-based, handles large files."""
import json, sys, subprocess, os
from pathlib import Path
from collections import Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from scraper.config import DATABASES

ENH = Path("data/enhanced")
STATE = json.load(open("data/extractor_state.json"))
state_counts = {k: len(v) for k, v in STATE.get("processed_doc_ids", {}).items()}
state_total = sum(state_counts.values())

TOP_FIELDS = ["doc_id", "url", "source_db", "court", "full_text"]
LLM_FIELDS = ["process_number", "court_name", "judge_name", "decision_date",
              "decision_type", "case_type", "summary", "legal_question",
              "decision_outcome", "ratio_decidendi", "legal_descriptors",
              "legislation_cited", "is_auj", "auj_number"]

print(f"STATE: {state_total:,} total docs across {len(state_counts)} databases")
for k, v in sorted(state_counts.items()):
    print(f"  {k}: {v:,}")

print(f"\n{'='*90}")

for d in DATABASES:
    short, db_key, label = d["short"], d["db"], d["label"]
    expected = d["approx_count"]
    enh_dir = ENH / short
    if not enh_dir.exists():
        print(f"\n{short} ({label}): ❌ NO ENHANCED DATA"); continue

    chunks = sorted(enh_dir.glob("chunk_*.jsonl"))
    if not chunks:
        print(f"\n{short} ({label}): ❌ NO CHUNK FILES"); continue

    # File-level info
    total_size = sum(f.stat().st_size for f in chunks)
    s_count = state_counts.get(db_key, 0)

    # Sample up to 500 docs from first chunk for quality analysis
    top_cnt = Counter(); llm_cnt = Counter()
    dates = []; auj_count = 0; sampled = 0
    seen_ids = set(); dup_count = 0
    parse_errors = 0

    for chunk in chunks:
        with open(chunk) as fh:
            for line in fh:
                if sampled >= 500: break
                sampled += 1
                stripped = line.strip()
                if not stripped: continue
                try:
                    doc = json.loads(stripped)
                except json.JSONDecodeError:
                    parse_errors += 1; continue

                did = doc.get("doc_id", "")
                if did in seen_ids: dup_count += 1
                seen_ids.add(did)

                for fld in TOP_FIELDS:
                    v = doc.get(fld)
                    if v is not None and v != "" and v != []: top_cnt[fld] += 1

                llm = doc.get("llm_extracted", {})
                for fld in LLM_FIELDS:
                    v = llm.get(fld)
                    if v is not None and v != "" and v != [] and v != {}: llm_cnt[fld] += 1

                if llm.get("is_auj"): auj_count += 1

                dt = llm.get("decision_date")
                if dt:
                    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
                        try: dates.append(datetime.strptime(str(dt), fmt)); break
                        except: pass
        if sampled >= 500: break

    print(f"\n{short} ({label})")
    print(f"  Files: {len(chunks)} chunks, {total_size/1e6:.0f} MB")
    print(f"  State: {s_count:,} / {expected:,} expected ({s_count/expected*100:.1f}%)")
    print(f"  Sampled: {sampled} docs | Parse errors: {parse_errors} | Dups in sample: {dup_count}")
    if dates:
        print(f"  Dates: {min(dates).strftime('%Y-%m-%d')} → {max(dates).strftime('%Y-%m-%d')} ({len(dates)}/{sampled} valid)")
    print(f"  AUJ in sample: {auj_count}/{sampled} ({auj_count/sampled*100:.1f}%)")

    # Field completeness
    low_fields = []
    for fld in TOP_FIELDS:
        pct = top_cnt.get(fld, 0) / sampled * 100
        if pct < 95: low_fields.append(f"{fld}={pct:.0f}%")
    for fld in LLM_FIELDS:
        pct = llm_cnt.get(fld, 0) / sampled * 100
        if pct < 80: low_fields.append(f"llm.{fld}={pct:.0f}%")

    if low_fields:
        print(f"  ⚠️  Low fill: {', '.join(low_fields)}")
    else:
        print(f"  ✅ All fields ≥80% filled")

print(f"\n{'='*90}")
print(f"GRAND TOTAL: {state_total:,} extracted | {state_total/sum(d['approx_count'] for d in DATABASES)*100:.1f}% coverage")
print(f"{'='*90}")
