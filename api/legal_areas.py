"""Canonical legal areas and resolution of free-text area filters.

`documents.legal_domain` is free text written by the extraction LLM (6,000+
distinct values, synonyms such as "Fiscal"/"Tributário", arbitrary ordering of
combined areas). scripts/canonicalize_legal_domains.py maps every raw value
onto the canonical areas below and stores the result in `legal_domain_map`;
the API resolves a user's filter string to canonical areas and filters through
that table. Shared by the API and the ETL so both use one vocabulary.
"""
from __future__ import annotations

import unicodedata

CANONICAL_AREAS: list[str] = [
    "Direito Administrativo",
    "Direito Tributário",
    "Direito Constitucional",
    "Direito Civil",
    "Direito Processual Civil",
    "Direito Penal",
    "Direito Processual Penal",
    "Direito Contraordenacional",
    "Direito do Trabalho",
    "Direito da Segurança Social",
    "Direito Comercial e Societário",
    "Direito da Insolvência",
    "Direito da Família e Menores",
    "Direito das Sucessões",
    "Direito do Arrendamento",
    "Direito dos Seguros",
    "Direito Bancário e Financeiro",
    "Direito do Consumo",
    "Direito do Urbanismo e Ordenamento do Território",
    "Direito do Ambiente",
    "Direito Registral e Notarial",
    "Direito da Propriedade Intelectual",
    "Direito da União Europeia e Internacional",
    "Direito dos Transportes",
    "Direito da Concorrência e Regulação",
]

# Synonyms users (and the raw data) use that are not substrings of a
# canonical name. Keys are normalized (lowercase, no accents).
ALIASES: dict[str, list[str]] = {
    "fiscal": ["Direito Tributário"],
    "impostos": ["Direito Tributário"],
    "aduaneiro": ["Direito Tributário"],
    "criminal": ["Direito Penal", "Direito Processual Penal"],
    "laboral": ["Direito do Trabalho"],
    "consumidor": ["Direito do Consumo"],
    "societario": ["Direito Comercial e Societário"],
    "empresas": ["Direito Comercial e Societário"],
    "falencia": ["Direito da Insolvência"],
    "contencioso administrativo": ["Direito Administrativo"],
    "obrigacoes": ["Direito Civil"],
    "contratos": ["Direito Civil"],
    "responsabilidade civil": ["Direito Civil"],
    "reais": ["Direito Civil"],
    "arrendamento urbano": ["Direito do Arrendamento"],
    "heranca": ["Direito das Sucessões"],
    "registos": ["Direito Registral e Notarial"],
    "notariado": ["Direito Registral e Notarial"],
    "bancario": ["Direito Bancário e Financeiro"],
    "valores mobiliarios": ["Direito Bancário e Financeiro"],
    "marcas": ["Direito da Propriedade Intelectual"],
    "patentes": ["Direito da Propriedade Intelectual"],
    "direito de autor": ["Direito da Propriedade Intelectual"],
    "europeu": ["Direito da União Europeia e Internacional"],
    "comunitario": ["Direito da União Europeia e Internacional"],
    "internacional": ["Direito da União Europeia e Internacional"],
    "estradal": ["Direito dos Transportes"],
    "rodoviario": ["Direito dos Transportes"],
    "ordenamento": ["Direito do Urbanismo e Ordenamento do Território"],
    "expropriacoes": ["Direito Administrativo"],
    "concorrencia": ["Direito da Concorrência e Regulação"],
    "regulacao": ["Direito da Concorrência e Regulação"],
    "contraordenacoes": ["Direito Contraordenacional"],
    "seguranca social": ["Direito da Segurança Social"],
    "previdencia": ["Direito da Segurança Social"],
    "menores": ["Direito da Família e Menores"],
}

_GENERIC_WORDS = {"direito", "do", "da", "dos", "das", "de", "e"}


def normalize(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


_NORMALIZED_AREAS = {normalize(area): area for area in CANONICAL_AREAS}


def resolve_legal_areas(text: str) -> list[str]:
    """Resolve a free-text filter ("insolvencia", "Fiscal", "penal") to
    canonical areas. Returns [] when nothing matches, in which case callers
    fall back to an accent-insensitive substring match on the raw column.

    A canonical area matches when the normalized input is a substring of its
    name ("penal" -> Penal + Processual Penal, mirroring the old ILIKE
    behaviour) or when an alias key occurs in the input ("direito fiscal").
    """
    query = normalize(text)
    if not query or query in _GENERIC_WORDS:
        return []
    areas = [area for norm, area in _NORMALIZED_AREAS.items() if query in norm]
    for alias, alias_areas in ALIASES.items():
        if alias in query:
            areas.extend(alias_areas)
    return list(dict.fromkeys(areas))
