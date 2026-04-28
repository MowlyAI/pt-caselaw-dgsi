"""Download embeddings from Supabase and build local hnswlib indexes (one-pass)."""
import json
import os
import time
from pathlib import Path

import httpx
import hnswlib
import numpy as np
from dotenv import load_dotenv

load_dotenv(".env.local")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
BATCH = 1000
FIELDS = ["embedding", "embedding_context", "embedding_ratio"]
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def parse_halfvec(raw):
    if raw is None:
        return None
    if isinstance(raw, list):
        return np.array(raw, dtype=np.float32)
    s = raw.strip("[]")
    return np.fromstring(s, sep=",", dtype=np.float32)


def fetch_all():
    """Cursor-based pagination via doc_id ordering. Fetches all 3 embedding fields at once."""
    cols = ",".join(["doc_id"] + FIELDS)
    rows = []
    last_id = ""
    page = 0
    t0 = time.time()
    while True:
        params = {"select": cols, "limit": BATCH, "order": "doc_id"}
        if last_id:
            params["doc_id"] = f"gt.{last_id}"
        r = httpx.get(
            f"{SUPABASE_URL}/rest/v1/documents",
            params=params,
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=60,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_id = batch[-1]["doc_id"]
        page += 1
        if page % 10 == 0:
            print(f"  {len(rows)} rows ({time.time()-t0:.1f}s)")
    print(f"  fetched {len(rows)} rows in {time.time()-t0:.1f}s")
    return rows


def build_all():
    print("\nDownloading all embeddings in one pass ...")
    rows = fetch_all()

    for field in FIELDS:
        print(f"\nBuilding index for {field} ...")
        ids = []
        vectors = []
        for row in rows:
            vec = parse_halfvec(row.get(field))
            if vec is not None and len(vec) == EMBEDDING_DIM:
                ids.append(row["doc_id"])
                vectors.append(vec)
        print(f"  valid vectors: {len(vectors)}")
        if not vectors:
            print("  skipping")
            continue

        data = np.stack(vectors)
        index = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
        index.init_index(max_elements=len(data), ef_construction=200, M=16)
        index.add_items(data, np.arange(len(data)))
        index.set_ef(40)

        idx_path = DATA_DIR / f"{field}.idx"
        map_path = DATA_DIR / f"{field}_ids.json"
        index.save_index(str(idx_path))
        with open(map_path, "w") as f:
            json.dump(ids, f)
        print(f"  saved {idx_path} ({idx_path.stat().st_size/1e6:.1f} MB)")


def main():
    build_all()
    print("\nDone.")


if __name__ == "__main__":
    main()
