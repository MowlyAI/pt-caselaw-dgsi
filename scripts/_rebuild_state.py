#!/usr/bin/env python3
"""Rebuild extractor_state.json from JSONL chunk files on disk.

Uses regex extraction instead of full JSON parsing for speed."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scraper.config import DATABASES

state = {'processed_doc_ids': {}}

# Build mapping from short name → db key using the authoritative config
db_map = {d['short']: d['db'] for d in DATABASES}
print(f'DB map from config: {db_map}')

for db_dir in sorted(Path('data/enhanced').iterdir()):
    if not db_dir.is_dir():
        continue

    db_key = db_map.get(db_dir.name, (db_dir.name.lower() + '.nsf'))
    chunk_files = sorted(db_dir.glob('chunk_*.jsonl'))
    if not chunk_files:
        continue

    print(f'Processing {db_dir.name} ({len(chunk_files)} chunks)...', flush=True)

    # Use grep to extract doc_ids fast (avoids full JSON parsing)
    # Pattern handles both "doc_id":"val" and "doc_id": "val"
    files_str = [str(f) for f in chunk_files]
    result = subprocess.run(
        ['grep', '-ohE', '"doc_id": *"[^"]+"'] + files_str,
        capture_output=True, text=True
    )
    doc_ids = set()
    for match in result.stdout.strip().split('\n'):
        if match:
            # Extract value: last quoted string
            parts = match.split('"')
            if len(parts) >= 4:
                doc_ids.add(parts[3])

    state['processed_doc_ids'][db_key] = list(doc_ids)
    print(f'  {db_key}: {len(doc_ids)} unique doc_ids', flush=True)

# Write state directly (no atomic replace needed for rebuild)
with open('data/extractor_state.json', 'w') as f:
    json.dump(state, f)
    f.flush()
    os.fsync(f.fileno())

total = sum(len(ids) for ids in state['processed_doc_ids'].values())
print(f'Wrote state: {total} total docs')
