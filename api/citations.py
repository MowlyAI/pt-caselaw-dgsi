"""Resolve cited decisions (metadata.jurisprudence_cited) to corpus documents.

The extractor stores each cited decision as free text: a process number as
written by the court, plus an optional court and date. Lookup goes through
the normalized process number (see `PROCESS_KEY_SQL`, backed by an
expression index), then court and date pick between candidates.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Optional

# Must stay byte-identical to the expression in
# migrations/20261018_documents_process_number_key_index.sql, or Postgres
# will not use the index.
PROCESS_KEY_SQL = "upper(regexp_replace(process_number, '\\s+', '', 'g'))"

_PREFIX_RE = re.compile(
    r"^(?:PROC(?:ESSO)?\.?|P\.|N\.?[º°O]\.?|N\.)+",
)
_WS_RE = re.compile(r"\s+")

# Court names as the extractor writes them, mapped to corpus court_short codes.
# Order matters: more specific names first.
_COURT_NAME_CODES: tuple[tuple[str, str], ...] = (
    ("supremo tribunal de justica", "STJ"),
    ("supremo tribunal administrativo", "STA"),
    ("tribunal constitucional", "TC"),
    ("tribunal dos conflitos", "TCON"),
    ("central administrativo sul", "TCAS"),
    ("central administrativo norte", "TCAN"),
    ("relacao do porto", "TRP"),
    ("relacao de lisboa", "TRL"),
    ("relacao de coimbra", "TRC"),
    ("relacao de evora", "TRE"),
    ("relacao de guimaraes", "TRG"),
    ("julgado de paz", "JP"),
)
_KNOWN_CODES = {code for _, code in _COURT_NAME_CODES}
_CODE_ALIASES = {"TCA SUL": "TCAS", "TCA NORTE": "TCAN", "TCASUL": "TCAS", "TCANORTE": "TCAN"}


def _strip_accents(value: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c)
    )


def process_key(raw: Optional[str]) -> Optional[str]:
    """Normalize a process number for lookup, or None when it is not one.

    "Proc. 157/12.8 TUGMR.G1.S1" -> "157/12.8TUGMR.G1.S1". Values without a
    digit (e.g. "Assento STJ") or too short to be unique return None.
    """
    if not raw:
        return None
    key = _WS_RE.sub("", raw).upper()
    key = _PREFIX_RE.sub("", key)
    if len(key) < 4 or len(key) > 80 or not any(c.isdigit() for c in key):
        return None
    return key


def court_code(abbreviation: Optional[str], name: Optional[str]) -> Optional[str]:
    """Map a cited court to a corpus court_short code, or None if unknown."""
    if abbreviation:
        abbr = abbreviation.strip().upper()
        abbr = _CODE_ALIASES.get(abbr, abbr)
        if abbr in _KNOWN_CODES:
            return abbr
    if name:
        folded = _strip_accents(name).lower()
        for needle, code in _COURT_NAME_CODES:
            if needle in folded:
                return code
    return None


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def pick_candidate(
    candidates: list[dict],
    court: Optional[str],
    decision_date: Optional[str],
) -> Optional[dict]:
    """Choose the cited document among rows sharing its process number.

    Each candidate has `court_short` and `decision_date` (date or None).
    A match must not contradict the citation: a different court AND a
    different date reject it. Between several, court then exact date
    decide; if still ambiguous nothing is returned rather than a guess.
    """
    if not candidates:
        return None
    cited_date = _parse_date(decision_date)

    def court_ok(c: dict) -> bool:
        return court is None or c.get("court_short") == court

    def date_ok(c: dict) -> bool:
        return cited_date is None or c.get("decision_date") == cited_date

    plausible = [c for c in candidates if court_ok(c) or date_ok(c)]
    if len(plausible) <= 1:
        return plausible[0] if plausible else None
    for narrowed in (
        [c for c in plausible if court_ok(c) and date_ok(c)],
        [c for c in plausible if court_ok(c)] if court else [],
        [c for c in plausible if date_ok(c)] if cited_date else [],
    ):
        if len(narrowed) == 1:
            return narrowed[0]
    return None
