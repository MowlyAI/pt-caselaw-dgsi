#!/usr/bin/env python3
"""
Deduplicate enhanced JSONL files.

For each doc_id, keeps the LAST occurrence (most recent extraction).
Writes to .dedup temp files, verifies counts, then swaps.

Only processes DBs that actually have duplicates: STJ, STA, TCON.
"""
import json
import os
import sys
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).parent.parent))
from scraper.config import DATABASES

ENH = Path("data/enhanced")
CHUNK_SIZE = 15_000  # lines per output chunk

DBS_TO_DEDUP = ["STJ", "STA", "TCON"]

# Expected unique counts from state file
state = json.load(open("data/extractor_state.json"))
state_counts = {k: len(v) for k, v in state.get("processed_doc_ids", {}).items()}

# Map short -> db_key
db_key_map = {d["short"]: d["db"] for d in DATABASES}


def extract_doc_id(line_bytes):
    """Fast doc_id extraction from a JSONL line (bytes)."""
    idx = line_bytes.find(b'"doc_id":')
    if idx < 0:
        return None
    rest = line_bytes[idx + 9:].lstrip()
    if not rest.startswith(b'"'):
        return None
    end = rest.index(b'"', 1)
    return rest[1:end].decode()


def dedup_database(short):
    """Deduplicate a single database's enhanced files."""
    db_key = db_key_map[short]
    expected_unique = state_counts.get(db_key, 0)
    enh_dir = ENH / short
    chunks = sorted(enh_dir.glob("chunk_*.jsonl"))

    if not chunks:
        print(f"  {short}: no chunk files found, skipping")
        return False

    # Pass 1: Read ALL lines, keeping last occurrence per doc_id
    print(f"  {short}: reading {len(chunks)} chunks...")
    docs = OrderedDict()  # doc_id -> line_bytes (keeps insertion order)
    total_lines = 0

    for chunk_path in chunks:
        with open(chunk_path, "rb") as f:
            for line in f:
                total_lines += 1
                did = extract_doc_id(line)
                if did is None:
                    print(f"    WARNING: no doc_id on line {total_lines} in {chunk_path.name}")
                    continue
                docs[did] = line  # overwrites earlier occurrence

    unique_count = len(docs)
    dup_count = total_lines - unique_count

    print(f"  {short}: {total_lines:,} lines -> {unique_count:,} unique ({dup_count:,} duplicates)")

    if dup_count == 0:
        print(f"  {short}: no duplicates, skipping")
        return True

    # Safety check: unique count must match state
    if unique_count != expected_unique:
        print(f"  ❌ {short}: unique count {unique_count:,} != state {expected_unique:,}. ABORTING.")
        return False

    print(f"  ✅ {short}: unique count matches state ({expected_unique:,})")

    # Pass 2: Write deduplicated chunks to temp files
    dedup_dir = enh_dir / "_dedup_tmp"
    dedup_dir.mkdir(exist_ok=True)

    chunk_idx = 0
    line_in_chunk = 0
    current_file = None
    written = 0

    for did, line_bytes in docs.items():
        if current_file is None or line_in_chunk >= CHUNK_SIZE:
            if current_file:
                current_file.close()
            fname = dedup_dir / f"chunk_{chunk_idx:04d}.jsonl"
            current_file = open(fname, "wb")
            chunk_idx += 1
            line_in_chunk = 0

        current_file.write(line_bytes)
        if not line_bytes.endswith(b"\n"):
            current_file.write(b"\n")
        line_in_chunk += 1
        written += 1

    if current_file:
        current_file.close()

    print(f"  {short}: wrote {written:,} lines to {chunk_idx} temp chunks")

    # Verify temp files
    verify_count = 0
    verify_ids = set()
    for f in sorted(dedup_dir.glob("chunk_*.jsonl")):
        with open(f, "rb") as fh:
            for line in fh:
                verify_count += 1
                did = extract_doc_id(line)
                if did:
                    verify_ids.add(did)

    if verify_count != unique_count or len(verify_ids) != unique_count:
        print(f"  ❌ {short}: verification FAILED. written={verify_count}, unique_ids={len(verify_ids)}, expected={unique_count}")
        return False

    print(f"  ✅ {short}: verification passed ({verify_count:,} lines, {len(verify_ids):,} unique IDs)")

    # Pass 3: Swap files - remove old chunks, move temp chunks into place
    for old_chunk in chunks:
        old_chunk.unlink()
        print(f"    removed {old_chunk.name}")

    for new_chunk in sorted(dedup_dir.glob("chunk_*.jsonl")):
        dest = enh_dir / new_chunk.name
        new_chunk.rename(dest)
        print(f"    moved {new_chunk.name} -> {dest.name}")

    dedup_dir.rmdir()
    print(f"  ✅ {short}: dedup complete")
    return True


print("=" * 60)
print("DEDUPLICATING ENHANCED JSONL FILES")
print("=" * 60)

results = {}
for short in DBS_TO_DEDUP:
    print(f"\nProcessing {short}...")
    results[short] = dedup_database(short)

print("\n" + "=" * 60)
print("RESULTS:")
for short, ok in results.items():
    print(f"  {short}: {'✅ SUCCESS' if ok else '❌ FAILED'}")
print("=" * 60)
