"""Simple progress poll for active index builds."""
import asyncio
import sys
import time

import asyncpg

PW = "*XH9B2RTSua23fR"
HOST = "aws-0-eu-west-1.pooler.supabase.com"
USER = "postgres.unorogsjlkimrehyndzs"


async def main():
    conn = await asyncpg.connect(
        host=HOST, port=5432, user=USER, password=PW,
        database="postgres", statement_cache_size=0,
    )
    print("Polling index build progress (Ctrl+C to stop)...")
    start = time.time()
    last_done = 0
    while True:
        rows = await conn.fetch("""
            SELECT now() - a.xact_start AS duration, p.phase,
                   p.blocks_done, p.blocks_total,
                   p.tuples_done, p.tuples_total
              FROM pg_stat_progress_create_index p
              JOIN pg_stat_activity a ON p.pid = a.pid
        """)
        if not rows:
            elapsed = time.time() - start
            print(f"\nNo active build. Done? (elapsed {elapsed:.0f}s)")
            break
        r = rows[0]
        dur = str(r["duration"]) if r["duration"] else "?"
        total = max(r["blocks_total"], 1)
        pct = r["blocks_done"] / total * 100
        rate = (r["blocks_done"] - last_done) / 10.0  # blocks per sec
        remaining = (total - r["blocks_done"]) / max(rate, 1)
        print(
            f"[{dur}] phase={r['phase']} "
            f"blocks={r['blocks_done']}/{total} ({pct:.1f}%) "
            f"tuples={r['tuples_done']} "
            f"rate={rate:.1f} blocks/s "
            f"eta={remaining:.0f}s",
            flush=True,
        )
        last_done = r["blocks_done"]
        await asyncio.sleep(10)
    await conn.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
