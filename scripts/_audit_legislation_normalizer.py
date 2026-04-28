"""Replay the legislation normalizer over the 100 already-extracted outputs
and audit how compound-article cases collapse to canonical singletons."""
from __future__ import annotations
import glob
import json
import re
from collections import Counter

from extractor.extractor import _normalize_legislation, _expand_article_ranges

CANON = re.compile(
    r"^\d+\.º(?:-[A-Za-z])?"                      # article number
    r"(?:,\s*n\.º\s+\d+)?"                        # optional single n.º
    r"(?:,\s*alínea\s+[a-z]\))?"                  # optional single alínea
    r"$"
)

files = sorted(glob.glob("data/eval/prod100/STJ_*.json"))
print(f"docs: {len(files)}")

raw_articles: list[str] = []
norm_articles: list[str] = []
raw_items_total = 0
norm_items_total = 0
for f in files:
    d = json.loads(open(f).read())
    items = d.get("legislation_cited") or []
    raw_items_total += len(items)
    for it in items:
        raw_articles.append((it.get("article") or "").strip())
    normalised = _normalize_legislation(items)
    norm_items_total += len(normalised)
    for it in normalised:
        norm_articles.append((it.get("article") or "").strip())

def pct_canonical(arts: list[str]) -> tuple[int, int, float]:
    ok = sum(1 for a in arts if CANON.match(a))
    return ok, len(arts), 100 * ok / max(1, len(arts))

ok_r, n_r, p_r = pct_canonical(raw_articles)
ok_n, n_n, p_n = pct_canonical(norm_articles)

print()
print("--- RAW (LLM output) ---")
print(f"  total entries:      {n_r}")
print(f"  canonical articles: {ok_r} ({p_r:.1f}%)")
print(f"  non-canonical:      {n_r - ok_r}")

print()
print("--- NORMALISED ---")
print(f"  total entries:      {n_n}")
print(f"  canonical articles: {ok_n} ({p_n:.1f}%)")
print(f"  non-canonical:      {n_n - ok_n}")
print(f"  Δ entries (expanded − deduped): {n_n - n_r:+d}")

# Residual failures to inspect.
bad_after = Counter(a for a in norm_articles if not CANON.match(a))
print()
print(f"--- RESIDUAL NON-CANONICAL ({len(bad_after)} distinct) ---")
for a, c in bad_after.most_common(25):
    print(f"  {c:3d}  {a!r}")

# Show concrete before/after on the originally-flagged patterns.
probe_patterns = [
    "527.º, n.ºs 1 e 2",
    "527.º, n.º 1 e 2",
    "801.º, n.ºs 1 e 2",
    "615.º, n.º 1, b)",
    "805.º, n.º 2, a)",
    "11.º, 12.º, 13.º, 19.º",
    "394.º, n.º 2, alíneas b) e f)",
    "119.º, n.ºs 1 e 3",
    "186.º, n.º 1 e 2, alínea d)",
    "189.º, n.ºs 1 e 2",
    "177.º, n.º 1, alíneas b), c) e 6",
    "213.º, n.º 1, alínea b), n.ºs 2 e 3",
    "8.º, n.ºs 2 e 6",
    "672.º, n.º 1, alíneas a) e b)",
    "41.º, n.º 2 e 3",
    "225.º, n.º 1, alínea b), n.º 2",
    "190.º, n.ºs 1 e 3",
    "180.º, n.ºs 2 a 4",
    "400.º, n.º 1, alíneas e) e f)",
    "256.º, n.ºs 1, alínea e) e 3",
]
print()
print("--- PROBES (before → after expansion) ---")
for p in probe_patterns:
    expanded = _expand_article_ranges(p)
    print(f"  {p!r:55s} → {expanded}")

# Law canonicalisation counts.
print()
print("--- distinct law names after normalisation ---")
laws = Counter()
for f in files:
    d = json.loads(open(f).read())
    items = _normalize_legislation(d.get("legislation_cited") or [])
    for it in items:
        laws[it["law"]] += 1
print(f"  distinct law strings: {len(laws)}")
for l, c in laws.most_common(15):
    print(f"  {c:4d}  {l}")
