import json
from pathlib import Path

seen = set()
dups = 0
total = 0
for f in sorted(Path('data/enhanced/STJ').glob('chunk_*.jsonl')):
    n = 0
    with open(f) as fh:
        for line in fh:
            total += 1
            n += 1
            try:
                d = json.loads(line)
                did = d.get('doc_id') or d.get('id')
                if did in seen:
                    dups += 1
                else:
                    seen.add(did)
            except Exception:
                pass
    print(f'{f.name}: {n} lines')
print(f'TOTAL: total_lines={total} unique_doc_ids={len(seen)} duplicates={dups}')
