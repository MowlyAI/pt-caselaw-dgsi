"""Judge rapporteur (relator) of a decision, and the key used to match it.

DGSI pages carry a structured "Relator:" field, kept in `documents.full_text`
as the label line followed by the name; the extractor copies it to
`metadata.judge_name`. The same judge is written with different case and
accents across courts and years ("JOSE CORREIA" / "José Correia"), so the
filter and the autocomplete match on `judge_key`, the name without accents,
punctuation or case. scripts/backfill_judges.py stores both in the
document_judges table.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")
# "Relator:" then the name, on the same line or the next one.
_RELATOR_LINE = re.compile(r"^\s*Relator\s*:[ \t]*(?:\n[ \t]*)?([^\n]*)", re.IGNORECASE | re.MULTILINE)
# Values that are another DGSI label rather than a name.
_LABEL = re.compile(r":\s*$|^(?:descritores|n[º°o]\s|data\s|votação|tribunal)", re.IGNORECASE)
MAX_NAME_LENGTH = 120


def judge_key(name: Any) -> str:
    """Accent-, case- and punctuation-insensitive form of a judge's name."""
    if not isinstance(name, str):
        return ""
    stripped = "".join(
        c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)
    )
    return _NON_ALNUM.sub(" ", stripped.upper()).strip()


def _clean(name: Any) -> Optional[str]:
    if not isinstance(name, str):
        return None
    name = " ".join(name.split())
    if len(name) < 3 or len(name) > MAX_NAME_LENGTH or _LABEL.search(name):
        return None
    return name if judge_key(name) else None


def relator_from_full_text(full_text: Any) -> Optional[str]:
    """The name in the page's "Relator:" field, or None when there is none."""
    if not isinstance(full_text, str):
        return None
    match = _RELATOR_LINE.search(full_text)
    return _clean(match.group(1)) if match else None


def judge_name(full_text: Any, metadata: Any) -> Optional[str]:
    """Relator of a decision: the page field first, the extracted one otherwise."""
    from_page = relator_from_full_text(full_text)
    if from_page:
        return from_page
    extracted = metadata.get("judge_name") if isinstance(metadata, dict) else None
    return _clean(extracted)


def query_tokens(q: Any, limit: int = 6) -> list[str]:
    """Normalised words of an autocomplete query, each matched as a word prefix."""
    return judge_key(q).split()[:limit]
