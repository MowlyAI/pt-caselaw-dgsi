#!/usr/bin/env python3
"""Part 2: Field-level quality check + duplicate analysis."""
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from scraper.config import DATABASES

ENH_DIR = Path("data/enhanced")

print("\n## 3. DUPLICATE DOC_ID ANALYSIS")
for db_info in DATABASES:
    short = db_info["short"]
    enh_dir = ENH_DIR / short
    if not enh_dir.exists():
        continue
    seen = Counter()
    total = 0
    for f in sorted(enh_dir.glob("chunk_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                total += 1
                try:
                    d = json.loads(line)
                    did = d.get("doc_id", "MISSING")
                    seen[did] += 1
                except:
                    pass
    dups = sum(1 for v in seen.values() if v > 1)
    dup_lines = sum(v - 1 for v in seen.values() if v > 1)
    unique = len(seen)
    if dups > 0:
        print(f"  {short}: {total:,} lines, {unique:,} unique, {dups:,} dup IDs ({dup_lines:,} extra lines)")
    else:
        print(f"  {short}: {total:,} lines, {unique:,} unique, ✅ no duplicates")

print("\n## 4. FIELD COMPLETENESS (sample first 200 docs per DB)")
EXPECTED_FIELDS = [
    "doc_id", "url", "case_number", "date", "court",
    "summary", "decision", "subject_area",
    "legislation_refs", "case_refs", "parties",
    "judges", "is_auj", "auj_number"
]

for db_info in DATABASES:
    short = db_info["short"]
    enh_dir = ENH_DIR / short
    if not enh_dir.exists():
        continue
    field_present = Counter()
    field_nonempty = Counter()
    sampled = 0
    for f in sorted(enh_dir.glob("chunk_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if sampled >= 200:
                    break
                try:
                    d = json.loads(line)
                    for fld in EXPECTED_FIELDS:
                        if fld in d:
                            field_present[fld] += 1
                            val = d[fld]
                            if val is not None and val != "" and val != [] and val != {}:
                                field_nonempty[fld] += 1
                except:
                    pass
                sampled += 1
        if sampled >= 200:
            break
    if sampled == 0:
        continue
    print(f"\n  {short} (sampled {sampled} docs):")
    for fld in EXPECTED_FIELDS:
        pres = field_present.get(fld, 0)
        nonempty = field_nonempty.get(fld, 0)
        pct = nonempty / sampled * 100 if sampled else 0
        flag = ""
        if pct < 50 and fld not in ("auj_number", "is_auj", "amounts_involved"):
            flag = " ⚠️"
        print(f"    {fld:<20} present={pres:>4}/{sampled}  nonempty={nonempty:>4}/{sampled} ({pct:5.1f}%){flag}")

print("\n## 5. AUJ DETECTION SUMMARY")
auj_total = 0
for db_info in DATABASES:
    short = db_info["short"]
    enh_dir = ENH_DIR / short
    if not enh_dir.exists():
        continue
    auj_count = 0
    total = 0
    for f in sorted(enh_dir.glob("chunk_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                total += 1
                try:
                    d = json.loads(line)
                    if d.get("is_auj"):
                        auj_count += 1
                except:
                    pass
    auj_total += auj_count
    pct = auj_count / total * 100 if total else 0
    print(f"  {short}: {auj_count:,} AUJs / {total:,} docs ({pct:.2f}%)")
print(f"  TOTAL AUJs: {auj_total:,}")
