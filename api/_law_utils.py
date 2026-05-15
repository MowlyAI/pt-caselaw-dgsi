"""Shared law-normalisation utilities used by both api/main.py and extractor/extractor.py.

Keeping these here (inside the api package) means the Docker image only needs to
ship api/ — extractor/ remains a pipeline-only module excluded by .dockerignore.
"""
from __future__ import annotations

import re

_LAW_ABBREV: dict[str, str] = {
    "cc": "Código Civil",
    "c.c.": "Código Civil",
    "cod. civil": "Código Civil",
    "cpc": "Código de Processo Civil",
    "ncpc": "Código de Processo Civil",
    "novo cpc": "Código de Processo Civil",
    "cp": "Código Penal",
    "cpp": "Código de Processo Penal",
    "ct": "Código do Trabalho",
    "cpt": "Código de Processo do Trabalho",
    "csc": "Código das Sociedades Comerciais",
    "crp": "Constituição da República Portuguesa",
    "c.r.p.": "Constituição da República Portuguesa",
    "cpa": "Código do Procedimento Administrativo",
    "cpta": "Código de Processo nos Tribunais Administrativos",
    "ccp": "Código dos Contratos Públicos",
    "rcp": "Regulamento das Custas Processuais",
    "ce": "Código da Estrada",
    "cod. estrada": "Código da Estrada",
    "cmc": "Código Comercial",
    "cod. comercial": "Código Comercial",
    "crc": "Código do Registo Criminal",
    "cnot": "Código do Notariado",
    "lgt": "Lei Geral Tributária",
    "lei tributária": "Lei Geral Tributária",
    "cep": "Código de Execução de Penas",
    "cjm": "Código de Justiça Militar",
}


def _canonicalize_law(raw: str) -> str:
    """Map common abbreviations to full Portuguese names and normalise
    statute/decree-law numbering to the canonical `Lei n.º X/YYYY` form."""
    if not raw:
        return raw
    s = raw.strip()
    # Drop trailing promulgation dates e.g. ", de 4 de julho".
    s = re.sub(r",\s*de\s+\d+\s+de\s+\w+(?:\s+de\s+\d+)?\s*$", "", s, flags=re.IGNORECASE)
    low = s.lower().strip()
    if low in _LAW_ABBREV:
        return _LAW_ABBREV[low]
    stripped = low.strip(". ")
    if stripped in _LAW_ABBREV:
        return _LAW_ABBREV[stripped]
    m = re.match(r"(?:decreto[\s-]?lei|dl)\s*(?:n[°º\.\s]*)?\s*(\d+(?:-[A-Za-z])?/\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"Decreto-Lei n.º {m.group(1)}"
    m = re.match(r"lei\s*(?:n[°º\.\s]*)?\s*(\d+(?:-[A-Za-z])?/\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"Lei n.º {m.group(1)}"
    return s


def _clean_article_text(s: str) -> str:
    """Harmonise ordinal markers, `alínea` spelling and whitespace."""
    if not s:
        return s
    s = s.strip()
    s = s.replace("°", "º")
    # bare "º" after an article number → ".º"
    s = re.sub(r"(\d+(?:-[A-Za-z])?)\s*º(?!\.)", r"\1.º", s)
    # "67-A" (no ordinal marker) → "67.º-A"
    s = re.sub(r"\b(\d+)(?=-[A-Za-z]\b)(?!\.º)", r"\1.º", s)
    # "al." → "alínea "
    s = re.sub(r"\bal\.\s*", "alínea ", s, flags=re.IGNORECASE)
    # ASCII "alinea" → "alínea"
    s = re.sub(r"\balinea(s?)\b", lambda m: "alínea" + m.group(1), s, flags=re.IGNORECASE)
    # "N.º" / "Nº" / "N°" / "n.ºs" → lowercase (preserve trailing "s")
    s = re.sub(r"\b[Nn]\.?\s*[º°](s?)\b", lambda m: "n.º" + (m.group(1) or ""), s)
    # strip an accidental "artigo" / "art." prefix
    s = re.sub(r"^(?:artigos?|arts?\.?)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(" .,;:")
