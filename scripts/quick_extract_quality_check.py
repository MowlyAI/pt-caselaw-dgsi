"""One-off quality check: rerun the 100-doc extraction (saving per-doc JSON)
so we can inspect fill rates, outliers and projected cost for the full corpus.

Writes per-doc JSON to data/eval/prod100/<doc_id>.json and a summary.json.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(".env.local")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extractor.extractor import extract_document, MODEL  # noqa: E402

RAW_DIR = Path("data/raw")
# Override via env: N, OUT_SUBDIR, CONCURRENCY, DB.
N = int(os.getenv("N", "100"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "20"))
DB = os.getenv("DB", "STJ")
OUT_DIR = Path("data/eval") / os.getenv("OUT_SUBDIR", f"prod{N}")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def iter_docs(db_short: str):
    db_dir = RAW_DIR / db_short
    for path in sorted(db_dir.glob("chunk_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


async def main():
    api_key = os.environ["OPENROUTER_API_KEY"]
    docs = list(itertools.islice(iter_docs(DB), N))
    print(f"model={MODEL} docs={len(docs)} concurrency={CONCURRENCY}")

    sem = asyncio.Semaphore(CONCURRENCY)
    usage: list = []
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[
            extract_document(client, d, api_key, sem, usage_sink=usage) for d in docs
        ], return_exceptions=True)
    elapsed = time.perf_counter() - t0

    ok = 0
    fill = {k: 0 for k in [
        "process_number", "decision_date", "judge_name", "court_name",
        "summary", "legal_question", "ratio_decidendi", "decision_outcome",
        "legal_domain", "procedural_type", "extraction_confidence",
        "semantic_search_query", "keywords_search_query",
    ]}
    list_counts = {k: [] for k in [
        "legislation_cited", "jurisprudence_cited", "doctrine_cited",
        "parties", "amounts_involved", "timeline_events", "legal_descriptors",
        "injuries", "insurance_companies",
    ]}
    anomalies: list[str] = []

    for doc, r in zip(docs, results):
        doc_id = doc.get("doc_id") or doc.get("url", "")[-40:]
        if isinstance(r, Exception) or r is None:
            anomalies.append(f"FAIL {doc_id}: {r!r}")
            continue
        ok += 1
        (OUT_DIR / f"{DB}_{doc_id}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for k in fill:
            v = r.get(k)
            if isinstance(v, str) and v.strip():
                fill[k] += 1
            elif v not in (None, "", [], {}):
                fill[k] += 1
        for k in list_counts:
            lst = r.get(k) or []
            if isinstance(lst, list):
                list_counts[k].append(len(lst))
                if len(lst) > 100:
                    anomalies.append(f"BIG {doc_id}.{k}={len(lst)}")
        # quick sanity checks
        sq = (r.get("semantic_search_query") or "").strip()
        if sq and not sq.endswith((".", "?")):
            anomalies.append(f"SQ no punct {doc_id}: {sq[:80]}")
        kw = (r.get("keywords_search_query") or "").strip()
        if kw:
            phrases = [p.strip() for p in kw.split(",")]
            if len(phrases) < 5:
                anomalies.append(f"KW too few phrases ({len(phrases)}) {doc_id}: {kw[:80]}")
            for ph in phrases:
                # must be lowercase ascii + digits + spaces only
                if re.search(r"[^a-z0-9 ]", ph):
                    anomalies.append(f"KW bad char {doc_id}: {ph!r}")
                    break
                toks = ph.split()
                if len(toks) < 2 or len(toks) > 6:
                    anomalies.append(f"KW phrase wrong length {len(toks)} {doc_id}: {ph!r}")
                    break

    p = sum(u.get("prompt_tokens", 0) for u in usage)
    cached = sum(
        (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        for u in usage
    )
    comp = sum(u.get("completion_tokens", 0) for u in usage)
    cost = sum(u.get("cost", 0.0) for u in usage)

    print(f"\nsuccess: {ok}/{len(docs)}  elapsed: {elapsed:.1f}s  "
          f"avg: {elapsed/len(docs):.2f}s/doc")
    print(f"prompt: {p:,}  cached: {cached:,} ({cached/p*100:.1f}%)  "
          f"completion: {comp:,}  cost: ${cost:.4f}")

    print("\n--- fill rates (scalar fields) ---")
    for k, c in fill.items():
        print(f"  {k:30s} {c:3d}/{ok}  ({c/ok*100:.0f}%)")

    print("\n--- list depth (mean / median / p95 / max) ---")
    for k, arr in list_counts.items():
        if arr:
            arr_s = sorted(arr)
            p95 = arr_s[int(len(arr_s) * 0.95)] if arr_s else 0
            print(f"  {k:25s} mean={statistics.mean(arr):5.1f} "
                  f"med={statistics.median(arr):4.1f} p95={p95:4d} "
                  f"max={max(arr):4d}")

    print(f"\n--- anomalies: {len(anomalies)} ---")
    for a in anomalies[:40]:
        print("  ", a)

    # --- legislation canonical-form audit ----------------------------------
    law_counter: Counter = Counter()
    art_bad: list[str] = []
    forbidden_laws = {  # canonical → list of wrong variants we want ZERO of
        "Código de Processo Civil": {"cpc", "ncpc", "novo cpc", "c.p.c."},
        "Código Civil": {"cc", "c.c.", "cód. civil"},
        "Código Penal": {"cp", "c.p."},
        "Código de Processo Penal": {"cpp", "c.p.p."},
        "Código do Trabalho": {"ct", "c.t."},
        "Constituição da República Portuguesa": {"crp", "c.r.p."},
        "Código das Sociedades Comerciais": {"csc"},
    }
    abbr_hits: Counter = Counter()
    art_regex = re.compile(r"^[0-9]+\.º(-[A-Z])?(, n\.º [0-9]+)?(, alínea [a-z]\))?$")
    for f in OUT_DIR.glob("*.json"):
        if f.name == "summary.json":
            continue
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for lc in (r.get("legislation_cited") or []):
            law = (lc.get("law") or "").strip()
            art = (lc.get("article") or "").strip()
            if law:
                law_counter[law] += 1
                low = law.lower().strip(" .,")
                for canonical, wrong_set in forbidden_laws.items():
                    if low in wrong_set:
                        abbr_hits[f"{law}  ->  should be: {canonical}"] += 1
                if "artigo" in low:
                    abbr_hits[f"LAW contains 'artigo': {law!r}"] += 1
            if art and not art_regex.match(art):
                art_bad.append(art)

    print(f"\n--- legislation canonical-form audit ---")
    print(f"distinct `law` strings: {len(law_counter)}")
    print("top 15 `law` values:")
    for law, n in law_counter.most_common(15):
        print(f"  {n:4d}  {law}")
    if abbr_hits:
        print(f"\nABBREVIATION HITS ({sum(abbr_hits.values())}):")
        for k, n in abbr_hits.most_common():
            print(f"  {n:3d}  {k}")
    else:
        print("\nno forbidden abbreviations found ✔")

    if art_bad:
        print(f"\nARTICLES NOT MATCHING CANONICAL REGEX ({len(art_bad)}):")
        for a in Counter(art_bad).most_common(20):
            print(f"  {a[1]:3d}  {a[0]!r}")
    else:
        print("all articles match canonical regex ✔")

    # projections
    docs_total = 387_780
    mean_secs = elapsed / ok if ok else 0
    per_doc_cost = cost / ok if ok else 0
    # With caching: after ~1k docs cached_rate should climb; use observed as floor
    projected_cost = per_doc_cost * docs_total
    # Wall-clock scales inversely with concurrency; at 20 → proportional
    wall_secs = mean_secs * docs_total
    print("\n--- projection for 387,780 docs ---")
    print(f"per-doc cost (observed): ${per_doc_cost:.5f}")
    print(f"total cost at observed rate: ${projected_cost:,.2f}")
    print(f"per-doc wall-clock (conc={CONCURRENCY}): {mean_secs:.2f}s")
    print(f"total wall-clock at conc=20: {wall_secs/3600:.1f} h")
    print(f"total wall-clock at conc=60: {wall_secs/3600/3:.1f} h")

    Path("data/eval/prod100/summary.json").write_text(json.dumps({
        "model": MODEL, "n": len(docs), "success": ok,
        "elapsed_s": elapsed, "cost_usd": cost,
        "prompt_tokens": p, "cached_tokens": cached, "completion_tokens": comp,
        "fill_rates": fill, "list_depth_mean": {
            k: statistics.mean(v) if v else 0 for k, v in list_counts.items()
        },
        "anomalies_count": len(anomalies),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
