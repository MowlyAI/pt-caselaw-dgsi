"""Permanent disability degree of a decision, as a number of points (0–100].

The extractor stores each injury's degree as free text in
`metadata.injuries[].disability_degree` ("19 pontos de défice funcional
permanente", "IPP de 17,55%", "7/100", "Grau 4/7", …). Searches such as
"invalidez superior a 15 pontos" need it as a number, so
scripts/backfill_disability_points.py parses it with `disability_points` into
the document_disability_points table, which the search filters read.

Civil "pontos" (défice funcional permanente) and labour-law IPP percentages are
both on a 0–100 scale and are stored as the same number.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# Degree segments on other scales or of other heads of damage: quantum doloris
# and dano estético (graded 1–7), repercussions on leisure or sexual activity,
# and temporary deficits.
_OTHER_SCALE = re.compile(
    r"\d\s*/\s*7\b|\bgraus?\s+\d|escala\s*(?:de\s*)?(?:1\s*[-a]\s*)?7\b|\bem\s+7\b|"
    r"quantum|est[ée]tic|repercuss|desportiv|lazer|sexual|tempor[áa]ri",
    re.IGNORECASE,
)
_OTHER_HEAD = re.compile(
    r"quantum|est[ée]tic|repercuss|desportiv|lazer|sexual|tempor[áa]ri",
    re.IGNORECASE,
)
# The first number written with a 0–100 unit: "19 pontos", "17,55%",
# "6 pontos percentuais", "7/100", "5 pontos em 100".
_POINTS = re.compile(
    r"(\d{1,3}(?:[.,]\d+)?)\s*(?:%|pontos?\b|pps?\b|p\.\s*p\.|/\s*100\b|em\s+100\b)",
    re.IGNORECASE,
)
_OWN_JOB = re.compile(r"\s*(?:\(|de\s+)?(?:para\s+a\s+)?(?:profiss|trabalho\s+habitual)", re.IGNORECASE)
# Labour-law coefficient written as a fraction of 1: "0,20", "0,173325 (IPP corrigida)".
_COEFFICIENT = re.compile(r"^\s*(0[.,]\d+)\b(?!\s*(?:%|pontos?\b))")


def injury_points(degree: Any, name: Any = None) -> Optional[float]:
    """Points of one injury's `disability_degree`, or None when it states none."""
    if not isinstance(degree, str) or not degree.strip():
        return None
    if isinstance(name, str) and _OTHER_HEAD.search(name):
        return None
    for segment in re.split(r"[;\n]", degree):
        if _OTHER_SCALE.search(segment):
            continue
        match = _POINTS.search(segment)
        if match and _OWN_JOB.match(segment, match.end()):
            # "100% para a profissão habitual" measures the own-job incapacity (IPATH).
            continue
        if match:
            value = float(match.group(1).replace(",", "."))
        elif coefficient := _COEFFICIENT.match(segment):
            value = float(coefficient.group(1).replace(",", ".")) * 100
        else:
            continue
        return value if 0 < value <= 100 else None
    return None


def disability_points(metadata: Any) -> Optional[float]:
    """Highest permanent disability degree among a decision's injuries.

    The global deficit is the largest figure; smaller ones are usually partial
    deficits of single injuries.
    """
    injuries = metadata.get("injuries") if isinstance(metadata, dict) else None
    if not isinstance(injuries, list):
        return None
    values = [
        points
        for injury in injuries
        if isinstance(injury, dict)
        and (points := injury_points(injury.get("disability_degree"), injury.get("name"))) is not None
    ]
    return max(values) if values else None
