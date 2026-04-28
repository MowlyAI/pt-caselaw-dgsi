#!/usr/bin/env python3
"""Test embedding API batch vs single reliability."""
import httpx, os, time
from dotenv import load_dotenv
load_dotenv(".env.local")
api_key = os.getenv("OPENROUTER_API_KEY")

MODELS = [
    ("qwen/qwen3-embedding-8b", 1024),
    ("google/gemini-embedding-001", 1024),
]

for model, dims in MODELS:
    print(f"\n=== {model} (dims={dims}) ===")
    # Test batch of 50
    texts = ["Responsabilidade civil extracontratual por facto ilicito, dano moral"] * 50
    start = time.time()
    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            json={"model": model, "input": texts, "dimensions": dims},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()["data"]
            print(f"  Batch 50: OK, {elapsed:.1f}s, {len(data)} embeddings")
        else:
            print(f"  Batch 50: ERROR {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  Batch 50: EXCEPTION after {elapsed:.1f}s: {e}")

    # Test single
    start = time.time()
    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            json={"model": model, "input": "test query", "dimensions": dims},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        elapsed = time.time() - start
        if resp.status_code == 200:
            print(f"  Single:   OK, {elapsed:.1f}s")
        else:
            print(f"  Single:   ERROR {resp.status_code}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  Single:   EXCEPTION after {elapsed:.1f}s: {e}")
