"""Audit the 500-doc sample for ways to deterministically detect
Acórdãos de Uniformização / Fixação de Jurisprudência (AUJ).

Checks multiple signals:
  1. procedural_type / case_type containing "uniformiz" or "fixa*jurispr"
  2. raw-document fields (Espécie, Meio Processual, Descritores) for AUJ hints
  3. text signatures in summary / ratio_decidendi
  4. jurisprudence_cited references (AUJ are often cited as "AUJ n.º X/YY")
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

SAMPLE = Path("data/eval/prod500")
RAW_DIR = Path("data/raw/STJ")


def norm(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


AUJ_RE = re.compile(r"(uniformiza[cç][aã]o|fixa[cç][aã]o de jurispr|fixar jurispr|uniformizador)")
AUJ_THESIS_RE = re.compile(
    r"(fixa[-\s]?se a seguinte jurispr|uniformiza[-\s]?se a jurispr|"
    r"acorda[-\s]se em fixar a seguinte|fixar jurispr\w*.{0,40} seguinte)"
)


def load_llm():
    for p in sorted(SAMPLE.glob("*.json")):
        try:
            yield p.stem, json.loads(p.read_text())
        except Exception:
            pass


HEADER_RE = re.compile(
    r"(?:Meio Processual|Decisão|Espécie|Descritores):\s*\n([^\n]+(?:\n[^\n:]+)*?)(?=\n[A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n:]{2,40}:\n|\Z)"
)


def parse_raw_headers(text: str) -> dict[str, str]:
    """Parse the header block that DGSI embeds at the top of full_text."""
    out: dict[str, str] = {}
    for key in ("Meio Processual", "Decisão", "Espécie", "Descritores"):
        m = re.search(
            rf"{re.escape(key)}:\s*\n([^\n]+(?:\n(?![A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n:]{{2,40}}:\n)[^\n]+)*)",
            text,
        )
        if m:
            out[key] = m.group(1).strip()
    return out


def load_raw_index(needed: set[str]) -> dict[str, dict]:
    """Load only raw docs whose doc_id is in `needed` (cheaper than reading all chunks)."""
    idx: dict[str, dict] = {}
    for path in sorted(RAW_DIR.glob("chunk_*.jsonl")):
        if len(idx) >= len(needed):
            break
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                did = d.get("doc_id") or ""
                if did in needed:
                    d["fields"] = parse_raw_headers(d.get("full_text", "")[:4000])
                    idx[did] = d
    return idx


def main():
    llm_by_id = dict(load_llm())
    print(f"loaded {len(llm_by_id)} LLM outputs")
    needed = {stem.split("_", 1)[-1] for stem in llm_by_id}
    raw = load_raw_index(needed)
    print(f"loaded {len(raw)} raw STJ docs for cross-reference")

    auj_by_type: list[str] = []
    auj_by_text: list[str] = []
    auj_by_raw: list[str] = []
    auj_by_any: set[str] = set()

    proc_type_counter: Counter = Counter()

    for doc_id, d in llm_by_id.items():
        did_core = doc_id.split("_", 1)[-1]  # strip the "stj_" prefix in eval dir
        pt = d.get("procedural_type") or ""
        ct = d.get("case_type") or ""
        combined = f"{pt} {ct}"
        if AUJ_RE.search(norm(combined)):
            auj_by_type.append(doc_id)
            auj_by_any.add(doc_id)
            proc_type_counter[f"{pt} / {ct}"] += 1

        # text signature
        blob = " ".join([
            d.get("summary") or "",
            d.get("ratio_decidendi") or "",
            d.get("legal_question") or "",
        ])
        if AUJ_THESIS_RE.search(norm(blob)):
            auj_by_text.append(doc_id)
            auj_by_any.add(doc_id)

        # raw-DGSI side — does Meio Processual or Espécie flag it?
        raw_d = raw.get(did_core)
        if raw_d:
            raw_fields = raw_d.get("fields") or {}
            meio = norm(raw_fields.get("Meio Processual"))
            especie = norm(raw_fields.get("Espécie"))
            decisao = norm(raw_fields.get("Decisão") or "")
            if AUJ_RE.search(f"{meio} {especie} {decisao}"):
                auj_by_raw.append(doc_id)
                auj_by_any.add(doc_id)

    print(f"\n=== AUJ detection signals ===")
    print(f"  via procedural_type/case_type regex : {len(auj_by_type)}")
    print(f"  via summary/ratio/legal_q text sig   : {len(auj_by_text)}")
    print(f"  via raw Meio Processual / Espécie    : {len(auj_by_raw)}")
    print(f"  union (any signal)                   : {len(auj_by_any)}")
    print()
    print("--- procedural_type / case_type for matches ---")
    for k, n in proc_type_counter.most_common():
        print(f"  {n:3d}  {k}")

    # where only the raw signal fires, show what fired
    only_raw = [d for d in auj_by_raw if d not in auj_by_type and d not in auj_by_text]
    print(f"\n--- raw-only matches ({len(only_raw)}) ---")
    for d in only_raw[:10]:
        did_core = d.split("_", 1)[-1]
        r = raw.get(did_core) or {}
        f = r.get("fields") or {}
        print(f"  {d[:40]}  Meio={f.get('Meio Processual')!r:50s}  Espécie={f.get('Espécie')!r}")

    # text-only matches (LLM missed it in procedural_type)
    only_text = [d for d in auj_by_text if d not in auj_by_type and d not in auj_by_raw]
    print(f"\n--- text-only matches ({len(only_text)}) ---")
    for d in only_text[:5]:
        r = llm_by_id[d]
        print(f"  {d[:40]}  pt={r.get('procedural_type')!r}  ct={r.get('case_type')!r}")
        print(f"    ratio snippet: {(r.get('ratio_decidendi') or '')[:200]}")

    # type-only matches (verify they really are AUJ)
    only_type = [d for d in auj_by_type if d not in auj_by_text and d not in auj_by_raw]
    print(f"\n--- type-only matches ({len(only_type)}) — verify ---")
    for d in only_type[:5]:
        r = llm_by_id[d]
        print(f"  {d[:40]}  pt={r.get('procedural_type')!r:40s}  ct={r.get('case_type')!r}")


if __name__ == "__main__":
    main()
