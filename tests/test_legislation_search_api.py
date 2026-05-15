"""Integration tests for the /search/legislation endpoint.

Tests that the LLM normalizer produces correct canonical citations
for a variety of clean, messy and edge-case inputs.

Run with the local server up on port 8000:
    python tests/test_legislation_search_api.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://localhost:8000"


def _post(endpoint: str, payload: dict, timeout: int = 75) -> dict:
    req = urllib.request.Request(
        f"{BASE}{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


CASES: list[tuple[str, str | None, str | None, str | None]] = [
    # (raw_input, expected_llm_canonicalized, expected_law, expected_article)
    # --- clean inputs (less common laws to keep DB query fast) ---
    ("CT art. 394", "CT art. 394", "Código do Trabalho", "394.º"),
    ("CP art. 150", "CP art. 150", "Código Penal", "150.º"),
    ("CPA art. 10", "CPA art. 10", "Código do Procedimento Administrativo", "10.º"),
    ("CPTA art. 5", "CPTA art. 5", "Código de Processo nos Tribunais Administrativos", "5.º"),
    ("CSC art. 20", "CSC art. 20", "Código das Sociedades Comerciais", "20.º"),
    ("CCP art. 15", "CCP art. 15", "Código dos Contratos Públicos", "15.º"),
    ("RCP art. 8", "RCP art. 8", "Regulamento das Custas Processuais", "8.º"),
    # --- messy / natural-language inputs ---
    ("artigo 2 co codigo do trabalho", "CT art. 2", "Código do Trabalho", "2.º"),
    ("artigo 394 do codigo do trabalho n 2", "CT art. 394, n.º 2", "Código do Trabalho", "394.º, n.º 2"),
    ("lei 65 de 2003 artigo 3", "Lei n.º 65/2003 art. 3", "Lei n.º 65/2003", "3.º"),
    ("decreto lei 15 93 artigo 26", "DL 15/93 art. 26", "Decreto-Lei n.º 15/93", "26.º"),
    # --- statute forms ---
    ("Lei n.º 38-A/2023 art. 5", "Lei n.º 38-A/2023 art. 5", "Lei n.º 38-A/2023", "5.º"),
    ("DL 15/93 art. 21", "DL 15/93 art. 21", "Decreto-Lei n.º 15/93", "21.º"),
    # --- law-only (no article) ---
    ("Código do Trabalho", "CT", "Código do Trabalho", None),
    ("Regulamento das Custas Processuais", "RCP", "Regulamento das Custas Processuais", None),
    # --- new abbreviations ---
    ("artigo 2 do codigo da estrada", "CE art. 2", "Código da Estrada", "2.º"),
]


def test_clean_and_messy_inputs() -> list[str]:
    failures: list[str] = []
    for raw, exp_llm, exp_law, exp_art in CASES:
        payload = {"articles": [raw], "match": "any", "limit": 1}
        try:
            resp = _post("/search/legislation", payload)
        except Exception as e:
            failures.append(f"FAIL  {raw!r}  →  exception: {e}")
            continue

        searched = resp.get("articles_searched", [])
        if not searched:
            failures.append(f"FAIL  {raw!r}  →  no articles_searched returned")
            continue

        art = searched[0]
        llm = art.get("llm_canonicalized")
        law = art.get("law")
        article = art.get("article")

        errs: list[str] = []
        if exp_llm is not None and llm != exp_llm:
            errs.append(f"llm_canonicalized: got {llm!r}, expected {exp_llm!r}")
        if law != exp_law:
            errs.append(f"law: got {law!r}, expected {exp_law!r}")
        if article != exp_art:
            errs.append(f"article: got {article!r}, expected {exp_art!r}")

        if errs:
            failures.append(f"FAIL  {raw!r}\n      " + "\n      ".join(errs))
        else:
            print(f"OK    {raw!r}  →  llm={llm!r}, law={law!r}, art={article!r}")

    return failures


def test_structured_input_bypasses_llm() -> list[str]:
    """Structured ArticleRef objects should bypass the LLM entirely."""
    failures: list[str] = []
    payload = {
        "articles": [{"law": "CT", "article": "394"}],
        "match": "any",
        "limit": 1,
    }
    resp = _post("/search/legislation", payload)
    searched = resp.get("articles_searched", [])
    if not searched:
        failures.append("FAIL  structured input  →  no articles_searched returned")
        return failures

    art = searched[0]
    if art.get("llm_canonicalized") is not None:
        failures.append(
            f"FAIL  structured input  →  llm_canonicalized should be None, got {art['llm_canonicalized']!r}"
        )
    else:
        print("OK    structured input  →  llm_canonicalized=None (bypassed)")
    return failures


def test_invalid_input_returns_422() -> list[str]:
    """Completely unrecognizable input should still return 422."""
    failures: list[str] = []
    for raw in ["banana", "artigo 1 do codigo dos valores mobiliarios"]:
        payload = {"articles": [raw], "match": "any", "limit": 1}
        try:
            _post("/search/legislation", payload)
            failures.append(f"FAIL  {raw!r}  →  expected 422, got 200")
        except urllib.error.HTTPError as e:
            if e.code == 422:
                print(f"OK    {raw!r}  →  422 as expected")
            else:
                failures.append(f"FAIL  {raw!r}  →  expected 422, got {e.code}")
    return failures


def test_very_common_law_returns_504_not_500() -> list[str]:
    """Very common laws (CPC, CRP, CC) may time out; we should get 504, not 500."""
    failures: list[str] = []
    for raw in ["CRP", "CPC art. 580", "CC art. 500"]:
        payload = {"articles": [raw], "match": "any", "limit": 1}
        try:
            _post("/search/legislation", payload)
            # If it succeeds quickly, that's fine too
            print(f"OK    {raw!r}  →  200 (query was fast)")
        except urllib.error.HTTPError as e:
            if e.code in (504,):
                print(f"OK    {raw!r}  →  504 as expected (query too slow)")
            else:
                failures.append(f"FAIL  {raw!r}  →  expected 200 or 504, got {e.code}")
    return failures


SEMANTIC_CASES: list[tuple[str, str | None, str | None]] = [
    # (query, expected_law_contains, expected_article_contains)
    # --- original problematic case ---
    ("artigo 2 do codigo da estrada", "Código da Estrada", "2"),
    # --- common laws with specific articles (previously 504) ---
    ("codigo civil artigo 342", "Código Civil", "342"),
    ("codigo civil artigo 334", "Código Civil", "334"),
    ("codigo civil artigo 483", "Código Civil", "483"),
    ("codigo penal artigo 150", "Código Penal", "150"),
    ("codigo do trabalho artigo 394", "Código do Trabalho", "394"),
    ("codigo do trabalho artigo 2", "Código do Trabalho", "2"),
    ("codigo comercial artigo 429", "Código Comercial", "429"),  # art. 429.º has 432 docs
    # --- messy / natural language ---
    ("artigo 394 do codigo do trabalho", "Código do Trabalho", "394"),
    ("artigo 580 do codigo processo civil", "Código de Processo Civil", "580"),
    ("artigo 69 do codigo penal", "Código Penal", "69"),
    # --- less common laws ---
    ("codigo das sociedades comerciais artigo 20", "Sociedades Comerciais", "20"),
    ("codigo contratos publicos artigo 15", "Contratos Públicos", "15"),
    ("regulamento custas processuais artigo 8", "Custas Processuais", "8"),
    ("codigo processo tribunais administrativos artigo 5", "Tribunais Administrativos", "5"),
    # --- law-only, less common ---
    ("codigo da estrada", "Código da Estrada", None),
    ("codigo comercial", "Código Comercial", None),
    ("codigo notariado", "Notariado", None),  # may be "Código de Notariado" or "Código do Notariado" in data
    ("lei geral tributaria", "Lei Geral Tributária", None),
]


def test_semantic_legislation_search() -> list[str]:
    """Semantic legislation search finds law via embeddings + LLM rerank."""
    failures: list[str] = []
    payload = {"q": "artigo 2 do codigo da estrada", "top_k": 20, "limit": 3}
    try:
        resp = _post("/search/legislation/semantic", payload, timeout=45)
    except Exception as e:
        failures.append(f"FAIL  semantic search  →  exception: {e}")
        return failures

    candidates = resp.get("candidates", [])
    if not candidates:
        failures.append("FAIL  semantic search  →  no candidates returned")
        return failures

    # At least one candidate should be Código da Estrada
    estrada_found = any(c.get("law") == "C\u00f3digo da Estrada" for c in candidates)
    if not estrada_found:
        laws = [c.get("law") for c in candidates[:5]]
        failures.append(f"FAIL  semantic search  →  expected C\u00f3digo da Estrada in top candidates, got {laws}")
    else:
        print("OK    semantic search  →  C\u00f3digo da Estrada found in candidates")

    # Results should be a list (may be empty if table is only partially populated)
    results = resp.get("results", [])
    if not isinstance(results, list):
        failures.append("FAIL  semantic search  →  results is not a list")
    else:
        print(f"OK    semantic search  →  {len(results)} results returned")

    return failures


def test_semantic_legislation_comprehensive() -> list[str]:
    """Run semantic search over many cases, checking for correct law, article, and results."""
    failures: list[str] = []
    for query, expected_law, expected_article in SEMANTIC_CASES:
        payload = {"q": query, "top_k": 20, "limit": 2}
        label = f"semantic({query!r})"
        try:
            resp = _post("/search/legislation/semantic", payload, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 504:
                # 504 is acceptable for law-only on very common laws
                if expected_article is None and expected_law and any(l in expected_law for l in ("Civil", "Penal", "Trabalho")):
                    print(f"OK    {label}  →  504 (law-only on common law, expected)")
                    continue
            failures.append(f"FAIL  {label}  →  HTTP {e.code}: {e.read().decode()[:120]}")
            continue
        except Exception as e:
            failures.append(f"FAIL  {label}  →  exception: {e}")
            continue

        candidates = resp.get("candidates", [])
        selected = [c for c in candidates if c.get("selected")]

        if not selected:
            failures.append(f"FAIL  {label}  →  no candidates selected by LLM")
            continue

        # Check that at least one selected candidate matches expected law
        law_match = any(expected_law in (c.get("law") or "") for c in selected)
        if not law_match:
            laws = [c.get("law") for c in selected]
            failures.append(
                f"FAIL  {label}  →  expected law containing {expected_law!r} in selected, got {laws}"
            )
            continue

        # If article expected, check at least one selected candidate matches
        if expected_article:
            art_match = any(expected_article in (c.get("article") or "") for c in selected)
            if not art_match:
                arts = [c.get("article") for c in selected]
                failures.append(
                    f"FAIL  {label}  →  expected article containing {expected_article!r} in selected, got {arts}"
                )
                continue

        # Verify returned documents actually cite the selected legislation
        results = resp.get("results", [])
        if not results:
            print(f"OK    {label}  →  law={selected[0]['law']!r}, article={selected[0].get('article')!r}, 0 results")
            continue

        # Verify that at least one returned result cites the expected legislation
        matched_doc = False
        for result in results:
            meta = result.get("metadata", {})
            leg_cited = meta.get("legislation_cited", [])
            for lc in leg_cited:
                lc_law = lc.get("law", "")
                lc_art = lc.get("article", "")
                if expected_law in lc_law:
                    if not expected_article or expected_article in lc_art:
                        matched_doc = True
                        break
            if matched_doc:
                break

        if not matched_doc:
            # Show citations from first result for debugging
            first_meta = results[0].get("metadata", {})
            first_leg = first_meta.get("legislation_cited", [])
            doc_laws = [(lc.get("law"), lc.get("article")) for lc in first_leg[:3]]
            failures.append(
                f"FAIL  {label}  →  no result cites expected legislation. First doc cited: {doc_laws}"
            )
            continue

        print(
            f"OK    {label}  →  law={selected[0]['law']!r}, "
            f"article={selected[0].get('article')!r}, {len(results)} results, doc verified"
        )

    return failures


def test_semantic_edge_cases() -> list[str]:
    """Edge cases for semantic search."""
    failures: list[str] = []

    # Empty query
    try:
        _post("/search/legislation/semantic", {"q": "", "limit": 1})
        failures.append("FAIL  semantic('')  →  expected 400, got 200")
    except urllib.error.HTTPError as e:
        if e.code == 400:
            print("OK    semantic('')  →  400 as expected")
        else:
            failures.append(f"FAIL  semantic('')  →  expected 400, got {e.code}")

    # Nonsense query
    try:
        resp = _post("/search/legislation/semantic", {"q": "banana sandwich", "limit": 1}, timeout=30)
        if resp.get("results"):
            print(f"OK    semantic('banana sandwich')  →  {len(resp['results'])} results (fallback)")
        else:
            print("OK    semantic('banana sandwich')  →  0 results (no relevant legislation)")
    except Exception as e:
        failures.append(f"FAIL  semantic('banana sandwich')  →  exception: {e}")

    return failures


if __name__ == "__main__":
    all_failures: list[str] = []
    all_failures.extend(test_clean_and_messy_inputs())
    all_failures.extend(test_structured_input_bypasses_llm())
    all_failures.extend(test_invalid_input_returns_422())
    all_failures.extend(test_very_common_law_returns_504_not_500())
    all_failures.extend(test_semantic_legislation_search())
    all_failures.extend(test_semantic_legislation_comprehensive())
    all_failures.extend(test_semantic_edge_cases())

    print()
    if all_failures:
        print(f"{len(all_failures)} FAILURE(S):")
        for f in all_failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)
