"""Print compact samples of one extraction per court to eyeball quality."""
import json
from pathlib import Path

OUT = Path("data/eval")
for court in ("STJ", "TRP", "TRL", "TCAS"):
    files = sorted(OUT.glob(f"{court}_*.json"))
    if not files:
        continue
    d = json.loads(files[0].read_text())
    e, r = d["extracted"], d["raw"]
    print(f"\n=== {court} / {files[0].name} ===")
    print(f"  raw full_text: {len(r['full_text']):,} chars")
    print(f"  url:         {r['url']}")
    print(f"  process:     {e['process_number']}")
    print(f"  court:       {e['court_name']}")
    print(f"  judge:       {e['judge_name']}")
    print(f"  date:        {e['decision_date']}")
    print(f"  legal_dom:   {e['legal_domain']}")
    print(f"  proc_type:   {e['procedural_type']}")
    print(f"  outcome:     {(e['decision_outcome'] or '')[:150]}")
    desc = e["legal_descriptors"] or []
    print(f"  descriptors: {desc[:5]} ({len(desc)} total)")
    print(f"  ratio:       {(e['ratio_decidendi'] or '')[:200]}")
    print(f"  semantic_q:  {e['semantic_search_query']}")
    print(f"  keywords:    {e['keywords_search_query']}")
    counts = (
        f"leg={len(e.get('legislation_cited') or [])} "
        f"jur={len(e.get('jurisprudence_cited') or [])} "
        f"doct={len(e.get('doctrine_cited') or [])} "
        f"parties={len(e.get('parties') or [])} "
        f"amounts={len(e.get('amounts_involved') or [])} "
        f"timeline={len(e.get('timeline_events') or [])}"
    )
    print(f"  counts:      {counts}")
    print(f"  evidence:    doc={e.get('documentary_evidence')} expert={e.get('expert_testimony')} "
          f"med={e.get('medical_evidence')} wit={e.get('witness_testimony')} "
          f"liab={e.get('liability_found')}")
    print(f"  confidence:  {e['extraction_confidence']}")
