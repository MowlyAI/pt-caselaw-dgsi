import json, glob, statistics, random
files = sorted(glob.glob('data/eval/prod100/STJ_*.json'))
counts = []
for f in files:
    d = json.loads(open(f).read())
    kw = d.get('keywords_search_query') or ''
    phrases = [p for p in kw.split(',') if p.strip()]
    counts.append(len(phrases))

print(f'docs: {len(counts)}')
print(f'phrases/doc: min={min(counts)} median={statistics.median(counts)} '
      f'mean={statistics.mean(counts):.1f} max={max(counts)}')
print(f'<5 phrases: {sum(1 for c in counts if c < 5)} docs')
print(f'<8 phrases: {sum(1 for c in counts if c < 8)} docs')

print()
print('--- random 5 outputs ---')
random.seed(1)
for f in random.sample(files, 5):
    d = json.loads(open(f).read())
    print(f'{f.split("/")[-1][:22]}:')
    print(f'  KW: {d.get("keywords_search_query")}')
    print(f'  SQ: {d.get("semantic_search_query")}')
    print()

print('--- legislation first doc ---')
d = json.loads(open(files[0]).read())
for l in (d.get('legislation_cited') or [])[:8]:
    print(f'  article={l.get("article")!r:35s}  law={l.get("law")!r:40s}  ctx={l.get("citation_context")!r}')
