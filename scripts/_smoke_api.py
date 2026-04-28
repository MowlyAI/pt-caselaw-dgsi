"""End-to-end smoke test for the locally-running API."""
import time
import urllib.parse
import urllib.request
import json


BASE = "http://127.0.0.1:8765"


def fetch(path, params=None, timeout=60):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = r.read()
    dt = (time.perf_counter() - t0) * 1000
    return r.status, dt, json.loads(body) if body else None


def show_results(label, code, dt, data, n_show=2):
    print(f"\n=== {label}  HTTP {code}  {dt:.0f} ms ===")
    if isinstance(data, dict) and "results" in data:
        print(f"  count={data['count']}  mode={data['mode']}  field={data['embedding_field']}")
        for r in data["results"][:n_show]:
            print(
                f"  - {r['doc_id'][:12]}.. {r['court_short']:5} "
                f"{r.get('decision_date','?')}  "
                f"sim={r.get('similarity')!s:>6}  "
                f"fts={r.get('fts_rank')!s:>6}  "
                f"hyb={r.get('hybrid_score')!s:>8}"
            )
            print(f"      summary: {(r.get('summary') or '')[:110]}")
    else:
        print(json.dumps(data, indent=2)[:400])


def main():
    code, dt, d = fetch("/health")
    show_results("/health", code, dt, d)

    code, dt, d = fetch("/stats")
    show_results("/stats", code, dt, d)

    queries = [
        "responsabilidade civil extracontratual do Estado",
        "despedimento sem justa causa indemnização",
        "acidente de trabalho nexo de causalidade",
    ]
    for q in queries:
        code, dt, d = fetch("/search/fts", {"q": q, "limit": 5})
        show_results(f"/search/fts q={q!r}", code, dt, d)

        code, dt, d = fetch("/search/semantic", {"q": q, "limit": 5})
        show_results(f"/search/semantic q={q!r}", code, dt, d)

        code, dt, d = fetch("/search", {"q": q, "limit": 5})
        show_results(f"/search (hybrid) q={q!r}", code, dt, d)

    # /document/{id}
    if d and d["results"]:
        doc_id = d["results"][0]["doc_id"]
        code, dt, doc = fetch(f"/document/{doc_id}")
        print(f"\n=== /document/{doc_id[:12]}..  HTTP {code}  {dt:.0f} ms ===")
        print(f"  url: {doc['url']}")
        print(f"  court: {doc['court_short']}  date: {doc['decision_date']}")
        print(f"  metadata keys: {list((doc.get('metadata') or {}).keys())[:8]}")


if __name__ == "__main__":
    main()
