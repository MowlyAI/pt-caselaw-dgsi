"""Audit the 100-doc sample to identify structured fields that would benefit
from deterministic normalization. Prints frequency tables for each candidate
field so we can decide which ones should get controlled vocabularies."""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SAMPLE = Path(os.environ.get("SAMPLE", "data/eval/prod500"))


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm(s: str | None) -> str:
    if not s:
        return ""
    return strip_accents(s).lower().strip()


def load():
    for p in sorted(SAMPLE.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            yield d.get("llm_extracted") or d
        except Exception:
            pass


def main():
    docs = list(load())
    print(f"loaded {len(docs)} extractions from {SAMPLE}")
    print()

    fields = [
        "decision_type",
        "case_type",
        "procedural_type",
        "legal_domain",
        "instance_level",
        "voting",
        "extraction_confidence",
        "decision_outcome",
    ]

    for f in fields:
        vals = [d.get(f) for d in docs if d.get(f)]
        print(f"=== {f}  ({len(vals)}/{len(docs)} filled) ===")
        # cluster by normalized key
        clusters: dict[str, Counter] = {}
        for v in vals:
            k = norm(v)
            clusters.setdefault(k, Counter())[v] += 1
        ordered = sorted(clusters.items(), key=lambda kv: -sum(kv[1].values()))
        print(f"  distinct canonical keys: {len(ordered)}")
        for k, variants in ordered[:20]:
            total = sum(variants.values())
            if len(variants) > 1:
                joined = " | ".join(f"{v!r}×{n}" for v, n in variants.most_common())
                print(f"    {total:3d}  {k!r:40s}  ⇢  {joined}")
            else:
                v = next(iter(variants))
                print(f"    {total:3d}  {v!r}")
        if len(ordered) > 20:
            print(f"    ... ({len(ordered) - 20} more)")
        print()

    # party roles/types (nested)
    print("=== party.role  (nested) ===")
    role_c: Counter = Counter()
    type_c: Counter = Counter()
    for d in docs:
        for p in (d.get("parties") or []):
            if p.get("role"):
                role_c[p["role"]] += 1
            if p.get("type"):
                type_c[p["type"]] += 1
    print(f"  distinct role values: {len(role_c)}")
    for v, n in role_c.most_common(25):
        print(f"    {n:3d}  {v!r}")
    print(f"  distinct type values: {len(type_c)}")
    for v, n in type_c.most_common(10):
        print(f"    {n:3d}  {v!r}")
    print()

    # decision_outcome patterns — try to pull a leading verb
    print("=== decision_outcome — leading verb pattern ===")
    verbs: Counter = Counter()
    for d in docs:
        o = d.get("decision_outcome") or ""
        m = re.match(r"\s*([A-Za-zÀ-ÿ\-]+(?:\s+[a-zÀ-ÿ]+)?)", o)
        if m:
            verbs[norm(m.group(1))] += 1
    for v, n in verbs.most_common(25):
        print(f"    {n:3d}  {v!r}")


if __name__ == "__main__":
    main()
