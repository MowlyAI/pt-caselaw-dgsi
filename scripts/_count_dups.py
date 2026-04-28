#!/usr/bin/env python3
"""Count duplicate doc_ids per enhanced file. Fast string-based extraction."""
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from scraper.config import DATABASES

ENH = Path("data/enhanced")

print("DUPLICATE COUNT PER DB AND FILE")
print("=" * 70)

for d in DATABASES:
    short = d["short"]
    enh_dir = ENH / short
    if not enh_dir.exists():
        continue

    db_seen = Counter()
    db_total = 0
    file_info = []

    for f in sorted(enh_dir.glob("chunk_*.jsonl")):
        file_seen = Counter()
        file_total = 0
        with open(f, "rb") as fh:
            for line in fh:
                file_total += 1
                db_total += 1
                idx = line.find(b'"doc_id":')
                if idx >= 0:
                    # Skip past "doc_id": and any whitespace/quote
                    rest = line[idx + 9:].lstrip()
                    if rest.startswith(b'"'):
                        start = 1
                        end = rest.index(b'"', start)
                        did = rest[start:end].decode()
                        file_seen[did] += 1
                        db_seen[did] += 1

        file_unique = len(file_seen)
        file_dups = sum(v - 1 for v in file_seen.values() if v > 1)
        file_info.append((f.name, file_total, file_unique, file_dups))

    db_unique = len(db_seen)
    db_dup_ids = sum(1 for v in db_seen.values() if v > 1)
    db_dup_lines = sum(v - 1 for v in db_seen.values() if v > 1)

    has_dups = db_dup_lines > 0
    print(f"\n{short}: {db_total:,} lines, {db_unique:,} unique, "
          f"{db_dup_lines:,} duplicate lines ({db_dup_ids:,} IDs repeated)"
          f" {'⚠️  NEEDS DEDUP' if has_dups else '✅ CLEAN'}")
    for fname, ft, fu, fd in file_info:
        flag = f" ({fd} intra-file dups)" if fd > 0 else ""
        print(f"    {fname}: {ft:,} lines, {fu:,} unique{flag}")

    if has_dups:
        # Show cross-file duplication
        cross_file = {}
        for f in sorted(enh_dir.glob("chunk_*.jsonl")):
            with open(f, "rb") as fh:
                for line in fh:
                    idx = line.find(b'"doc_id":')
                    if idx >= 0:
                        rest = line[idx + 9:].lstrip()
                        if rest.startswith(b'"'):
                            end = rest.index(b'"', 1)
                            did = rest[1:end].decode()
                        else:
                            continue
                        if did not in cross_file:
                            cross_file[did] = []
                        cross_file[did].append(f.name)
        multi_file = {k: v for k, v in cross_file.items() if len(set(v)) > 1}
        same_file_dups = db_dup_ids - len(multi_file)
        print(f"    Cross-file duplicates: {len(multi_file):,} IDs appear in multiple chunks")
        if same_file_dups > 0:
            print(f"    Same-file duplicates: {same_file_dups:,} IDs repeated within a single chunk")
