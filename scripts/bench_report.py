"""Print full (non-truncated) benchmark scorecard from summary.json."""
import json
from pathlib import Path

s = json.loads(Path("data/eval/benchmark/summary.json").read_text())
for r in sorted(s, key=lambda x: x["cost_usd"]):
    sc = r["scorecard"]
    print(f"\n=== {r['model']} ===")
    print(f"  ok:           {sc['ok']}/{sc['total']}")
    print(f"  elapsed:      {r['elapsed_s']:.1f}s")
    print(f"  cost (OR):    ${r['cost_usd']:.4f}")
    if sc["ok"]:
        print(f"  proj 387k:    ${r['cost_usd']/sc['ok']*387_780:,.0f}")
    print(f"  prompt tok:   {r['prompt_tokens']:,}")
    print(f"  completion:   {r['completion_tokens']:,}")
    print(f"  cached tok:   {r['cached']:,}")
    print(f"  dates_bad:    {sc['dates_invalid']}")
    print(f"  enum_bad:     {sc['bad_citation_ctx']}")
    print(f"  non_pt_query: {sc['non_pt_query']}")
    print(f"  confidence:   {dict(sc['confidence'])}")
    print(f"  fill_rates:")
    for k, v in sc["fill_rate"].items():
        marker = " " if v >= 0.95 else ("!" if v < 0.5 else "~")
        print(f"    {marker} {k:26s} {v*100:5.1f}%")
    print(f"  list_avg_lens:")
    for k, v in sc["list_avg"].items():
        print(f"      {k:26s} {v:6.1f}")
