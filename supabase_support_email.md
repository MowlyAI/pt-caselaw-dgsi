**Subject:** Re: Pro Plan — HNSW pgvector indexes completely unused by planner; 75-second sequential scans

Hi Max,

Quick follow-up to close the loop on this one: **the issue is resolved** and the HNSW indexes are now valid, used by the planner, and serving sub-second queries. Nothing further needed from your side, but I wanted to share the root cause and the recipe that worked, in case it's useful for other Pro-tier customers hitting the same symptoms.

---

## Root cause

The planner was never the problem. All three HNSW indexes were silently **invalid** (`indisvalid = false` in `pg_index`), even though `pg_indexes` listed them and the original `CREATE INDEX CONCURRENTLY` commands had returned without error. Postgres never considers an invalid index, so it correctly fell back to a parallel sequential scan — that's why `idx_scan` was permanently zero on the HNSW indexes while the GIN `fts` index on the same table worked fine.

The `CONCURRENTLY` builds had been failing partway through on Micro/Small compute (likely a combination of the 500 IOPS ceiling, 2 GB RAM, and `/dev/shm` limits when parallel maintenance workers were used), but failing in a way that left a half-built index marked invalid rather than raising a visible error to the client.

---

## What fixed it

1. **Scaled compute to Medium (4 GB RAM)** — temporarily, for the rebuild only.
2. **Dropped the invalid indexes** and rebuilt them with **non-concurrent** `CREATE INDEX` (acceptable here because the table is read-only during the build window).
3. Set per-session GUCs for the build:
   ```sql
   SET maintenance_work_mem = '1536MB';
   SET max_parallel_maintenance_workers = 0;
   ```
   Disabling parallel maintenance workers was important — with parallelism enabled the builds were hitting `/dev/shm` limits.
4. Built the three indexes sequentially (one at a time), each taking ~15–20 minutes.

After the rebuild:

| Index | Status | Plan | Warm latency |
|---|---|---|---|
| `idx_documents_embedding` | valid | `Index Scan using idx_documents_embedding` | ~98 ms |
| `idx_documents_embedding_context` | valid | `Index Scan using idx_documents_embedding_context` | ~98 ms |
| `idx_documents_embedding_ratio` | valid | `Index Scan using idx_documents_embedding_ratio` | ~102 ms |

`EXPLAIN ANALYZE` confirms execution time under 1 ms; the rest is network round-trip.

---

## One small piece of feedback

The only thing that was genuinely confusing during debugging was that `pg_indexes` happily lists invalid indexes with no indication of their state — you have to join against `pg_index.indisvalid` to see the truth. If a future Studio release could surface index validity in the index list (or warn when an `idx_scan = 0` index is invalid), it would have saved a couple of days here. Not a bug, just a nice-to-have.

---

Thanks again for the earlier pointers on the session pooler and on tuning the build — they put me on the right track even though the underlying cause turned out to be different. Feel free to close the ticket.

Best,
Francisco Costa
