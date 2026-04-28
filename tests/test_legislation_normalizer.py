"""Tests for the deterministic legislation post-processor.

Run with either pytest or as a plain script:
    PYTHONPATH=. python tests/test_legislation_normalizer.py
    PYTHONPATH=. pytest tests/test_legislation_normalizer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractor.extractor import (  # noqa: E402
    _canonicalize_law,
    _clean_article_text,
    _expand_article_ranges,
    _normalize_legislation,
)


# ---------------------------------------------------------------------------
# _expand_article_ranges
# ---------------------------------------------------------------------------

EXPAND_CASES: list[tuple[str, list[str]]] = [
    # canonical singletons pass through unchanged
    ("394.º", ["394.º"]),
    ("394.º, n.º 1", ["394.º, n.º 1"]),
    ("394.º, n.º 1, alínea a)", ["394.º, n.º 1, alínea a)"]),
    ("5.º-A", ["5.º-A"]),
    # "n.ºs X e Y" compound → split
    ("527.º, n.ºs 1 e 2", ["527.º, n.º 1", "527.º, n.º 2"]),
    ("801.º, n.ºs 1 e 2", ["801.º, n.º 1", "801.º, n.º 2"]),
    ("119.º, n.ºs 1 e 3", ["119.º, n.º 1", "119.º, n.º 3"]),
    # "n.º X e Y" (missing plural "s") also split
    ("527.º, n.º 1 e 2", ["527.º, n.º 1", "527.º, n.º 2"]),
    ("41.º, n.º 2 e 3", ["41.º, n.º 2", "41.º, n.º 3"]),
    # "alíneas X) e Y)" → split
    ("672.º, n.º 1, alíneas a) e b)",
     ["672.º, n.º 1, alínea a)", "672.º, n.º 1, alínea b)"]),
    ("394.º, n.º 2, alíneas b) e f)",
     ["394.º, n.º 2, alínea b)", "394.º, n.º 2, alínea f)"]),
    # implicit alínea (letter+paren without the word "alínea")
    ("615.º, n.º 1, b)", ["615.º, n.º 1, alínea b)"]),
    ("805.º, n.º 2, a)", ["805.º, n.º 2, alínea a)"]),
    # article-list
    ("11.º, 12.º, 13.º, 19.º", ["11.º", "12.º", "13.º", "19.º"]),
    ("394.º e 395.º", ["394.º", "395.º"]),
    # article-range
    ("234.º a 240.º", [
        "234.º", "235.º", "236.º", "237.º", "238.º", "239.º", "240.º",
    ]),
    # n.º range
    ("180.º, n.ºs 2 a 4",
     ["180.º, n.º 2", "180.º, n.º 3", "180.º, n.º 4"]),
    # cartesian product of n.º × alínea
    ("186.º, n.º 1 e 2, alínea d)",
     ["186.º, n.º 1, alínea d)", "186.º, n.º 2, alínea d)"]),
    # bare-ordinal repair ("67-A" with no .º) + implicit alínea
    ("67-A, alínea b)", ["67.º-A, alínea b)"]),
    # the "article" field receives an accidental "artigo" prefix
    ("artigo 643.º", ["643.º"]),
    # degree sign instead of masculine ordinal
    ("394°", ["394.º"]),
    # empty / null passthrough
    ("", []),
]


def run_expand_cases() -> None:
    for raw, expected in EXPAND_CASES:
        got = _expand_article_ranges(raw)
        assert got == expected, f"expand({raw!r}): expected {expected!r}, got {got!r}"


# ---------------------------------------------------------------------------
# _canonicalize_law
# ---------------------------------------------------------------------------

LAW_CASES: list[tuple[str, str]] = [
    # canonical pass-through
    ("Código de Processo Civil", "Código de Processo Civil"),
    ("Código Civil", "Código Civil"),
    # forbidden abbreviations → canonical
    ("CPC", "Código de Processo Civil"),
    ("ncpc", "Código de Processo Civil"),
    ("novo cpc", "Código de Processo Civil"),
    ("CC", "Código Civil"),
    ("c.c.", "Código Civil"),
    ("CP", "Código Penal"),
    ("CPP", "Código de Processo Penal"),
    ("CRP", "Constituição da República Portuguesa"),
    ("RCP", "Regulamento das Custas Processuais"),
    # statute numbering variants
    ("Lei nº 65/2003", "Lei n.º 65/2003"),
    ("Lei 65/2003", "Lei n.º 65/2003"),
    ("Lei n.º 65/2003", "Lei n.º 65/2003"),
    ("Decreto-Lei 15/93", "Decreto-Lei n.º 15/93"),
    ("DL 15/93", "Decreto-Lei n.º 15/93"),
    ("Decreto-Lei n.º 15/93", "Decreto-Lei n.º 15/93"),
    # trailing promulgation date gets stripped
    ("Lei n.º 23/2007, de 4 de julho", "Lei n.º 23/2007"),
    # unknown full-form law → unchanged
    ("Código da Estrada", "Código da Estrada"),
]


def run_law_cases() -> None:
    for raw, expected in LAW_CASES:
        got = _canonicalize_law(raw)
        assert got == expected, f"canon({raw!r}): expected {expected!r}, got {got!r}"


# ---------------------------------------------------------------------------
# _normalize_legislation — end-to-end dedup + priority
# ---------------------------------------------------------------------------

def test_dedup_priority_keeps_highest_ranked_context() -> None:
    items = [
        {"article": "615.º", "law": "CPC", "citation_context": "referencing"},
        {"article": "615.º", "law": "Código de Processo Civil", "citation_context": "supporting"},
    ]
    out = _normalize_legislation(items)
    assert len(out) == 1
    assert out[0] == {
        "article": "615.º",
        "law": "Código de Processo Civil",
        "citation_context": "supporting",
    }


def test_compound_entry_is_expanded_and_deduped() -> None:
    items = [
        {"article": "672.º, n.º 1, alíneas a) e b)", "law": "CPC",
         "citation_context": "referencing"},
        {"article": "672.º, n.º 1, alínea a)", "law": "Código de Processo Civil",
         "citation_context": "supporting"},
    ]
    out = _normalize_legislation(items)
    keys = sorted((r["article"], r["law"], r["citation_context"]) for r in out)
    assert keys == [
        ("672.º, n.º 1, alínea a)", "Código de Processo Civil", "supporting"),
        ("672.º, n.º 1, alínea b)", "Código de Processo Civil", "referencing"),
    ]


def test_invalid_context_falls_back_to_referencing() -> None:
    out = _normalize_legislation([
        {"article": "1.º", "law": "CC", "citation_context": "bogus"},
    ])
    assert out == [{"article": "1.º", "law": "Código Civil",
                    "citation_context": "referencing"}]


def test_clean_article_text_normalises_common_noise() -> None:
    # spacing, capitalisation, "al." → "alínea", "Nº" → "n.º"
    assert _clean_article_text("  Artigo  394º , Nº 2 , al. b)  ") == \
        "394.º , n.º 2 , alínea b)"


def _run_all() -> None:
    run_expand_cases()
    run_law_cases()
    test_dedup_priority_keeps_highest_ranked_context()
    test_compound_entry_is_expanded_and_deduped()
    test_invalid_context_falls_back_to_referencing()
    test_clean_article_text_normalises_common_noise()
    print(f"OK — {len(EXPAND_CASES) + len(LAW_CASES) + 4} assertions passed.")


if __name__ == "__main__":
    _run_all()
