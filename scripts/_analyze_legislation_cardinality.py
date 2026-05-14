"""Analyze legislation cardinality across all enhanced data."""
import glob, json, re
from collections import Counter
from extractor.extractor import _normalize_legislation

CANON = re.compile(
    r"^\d+\.º(?:-[A-Za-z])?"
    r"(?:,\s*n\.º\s+\d+)?"
    r"(?:,\s*alínea\s+[a-z]\))?"
    r"$"
)

files = sorted(glob.glob("data/enhanced/*/*.jsonl"))
print(f"Total files: {len(files)}")

total_docs = 0
total_raw_items = 0
total_norm_items = 0
raw_articles = []
norm_articles = []
laws = Counter()
all_law_names = Counter()
article_law_pairs = Counter()

for f in files:
    with open(f) as fh:
        for line in fh:
            total_docs += 1
            d = json.loads(line)
            items = d.get("llm_extracted", {}).get("legislation_cited") or []
            total_raw_items += len(items)
            for it in items:
                raw_articles.append((it.get("article") or "").strip())
                all_law_names[(it.get("law") or "").strip()] += 1

            normalised = _normalize_legislation(items)
            total_norm_items += len(normalised)
            for it in normalised:
                art = (it.get("article") or "").strip()
                law = (it.get("law") or "").strip()
                norm_articles.append(art)
                laws[law] += 1
                article_law_pairs[(art, law)] += 1

print(f"Total docs: {total_docs}")
print(f"Total raw legislation entries: {total_raw_items}")
print(f"Total normalized legislation entries: {total_norm_items}")
print()

def pct_canonical(arts):
    ok = sum(1 for a in arts if CANON.match(a))
    return ok, len(arts), 100 * ok / max(1, len(arts))

ok_r, n_r, p_r = pct_canonical(raw_articles)
ok_n, n_n, p_n = pct_canonical(norm_articles)

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
print()

bad_after = Counter(a for a in norm_articles if not CANON.match(a))
print(f"--- RESIDUAL NON-CANONICAL ({len(bad_after)} distinct) ---")
for a, c in bad_after.most_common(25):
    print(f"  {c:5d}  {a!r}")

print()
print(f"--- Distinct law names after normalisation: {len(laws)} ---")
for l, c in laws.most_common(20):
    print(f"  {c:6d}  {l}")

print()
print(f"--- Distinct raw law strings: {len(all_law_names)} ---")
for l, c in all_law_names.most_common(20):
    print(f"  {c:6d}  {l!r}")

print()
print(f"--- Distinct (article, law) pairs after normalization: {len(article_law_pairs)} ---")
for pair, c in article_law_pairs.most_common(20):
    print(f"  {c:5d}  {pair}")
