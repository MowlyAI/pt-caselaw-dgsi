"""
Concatenate split JSONL shards into a single file per database.

For every directory `data/<stage>/<court>/` containing `chunk_*.jsonl` files,
produces `data/<stage>/<court>.jsonl` by concatenating shards in order.

Keeps every individual shard under 100MB on disk (GitHub / Render constraint)
while still allowing a single consolidated file at build/runtime.

Usage:
    python scripts/concat_chunks.py [root_dir]
        default root_dir = "data"
"""
from __future__ import annotations

import sys
from pathlib import Path


def concat_chunks(root: Path) -> int:
    if not root.exists():
        return 0

    merged = 0
    for chunk_dir in sorted(root.rglob("*")):
        if not chunk_dir.is_dir():
            continue
        shards = sorted(chunk_dir.glob("chunk_*.jsonl"))
        if not shards:
            continue

        target = chunk_dir.with_suffix(".jsonl")
        total_bytes = 0
        with open(target, "wb") as out:
            for shard in shards:
                with open(shard, "rb") as src:
                    while True:
                        buf = src.read(1024 * 1024)
                        if not buf:
                            break
                        out.write(buf)
                        total_bytes += len(buf)
        merged += 1
        print(f"[concat] {chunk_dir} -> {target} ({total_bytes/1_048_576:.1f} MB from {len(shards)} shards)")

    return merged


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    count = concat_chunks(root)
    print(f"[concat] merged {count} database(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
