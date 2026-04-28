"""Deep-dive comparison across models on the same doc + check anomalies."""
import json
from pathlib import Path

BASE = Path("data/eval/benchmark")
MODELS = [
    "xiaomi__mimo-v2-flash",
    "google__gemini-3.1-flash-lite-preview",
    "google__gemini-2.5-flash-lite",
    "openai__gpt-5-nano",
    "qwen__qwen3.5-flash-02-23",
    "openai__gpt-4.1-nano",
]

# 1. Check gemini-2.5 anomaly: find docs with >100 legislation items
print("=== gemini-2.5 legislation_cited list sizes per doc ===")
g25 = BASE / "google__gemini-2.5-flash-lite"
for f in sorted(g25.glob("*.json")):
    try:
        d = json.loads(f.read_text())
        n = len(d.get("legislation_cited") or [])
        flag = " !!" if n > 50 else ""
        print(f"  {f.name}: {n} items{flag}")
    except Exception as e:
        print(f"  {f.name}: PARSE ERR {e}")

# 2. Pick one shared doc (STJ first doc) and show per-model summary + ratio
print("\n\n=== Same doc side-by-side (STJ first available) ===")
reference = sorted((BASE / MODELS[0]).glob("STJ_*.json"))
if reference:
    target_name = reference[0].name
    print(f"Target doc: {target_name}\n")
    for m in MODELS:
        p = BASE / m / target_name
        if not p.exists():
            print(f"--- {m}: MISSING ---\n"); continue
        d = json.loads(p.read_text())
        print(f"--- {m} ---")
        print(f"  process:     {d.get('process_number')}")
        print(f"  legal_dom:   {d.get('legal_domain')}")
        print(f"  proc_type:   {d.get('procedural_type')}")
        print(f"  confidence:  {d.get('extraction_confidence')}")
        print(f"  counts: leg={len(d.get('legislation_cited') or [])} "
              f"jur={len(d.get('jurisprudence_cited') or [])} "
              f"doct={len(d.get('doctrine_cited') or [])} "
              f"parties={len(d.get('parties') or [])} "
              f"amounts={len(d.get('amounts_involved') or [])} "
              f"timeline={len(d.get('timeline_events') or [])}")
        s = d.get("summary") or ""
        print(f"  summary[:200]: {s[:200]}")
        r = d.get("ratio_decidendi") or ""
        print(f"  ratio[:200]:   {r[:200]}")
        q = d.get("semantic_search_query") or ""
        print(f"  semantic_q:  {q[:200]}")
        print()

# 3. Show a specific offending legislation_cited entry for gemini-2.5
print("\n=== gemini-2.5 sample legislation entries (first doc, first 3) ===")
first_g25 = sorted(g25.glob("*.json"))[0]
d = json.loads(first_g25.read_text())
for i, leg in enumerate((d.get("legislation_cited") or [])[:3]):
    print(f"  [{i}] {json.dumps(leg, ensure_ascii=False)[:300]}")
