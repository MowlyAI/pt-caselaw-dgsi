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


# The document UNID is the last 32-hex segment (after the view id, if any).
_UNID_RE = re.compile(r"dgsi\.pt/[a-z0-9]+\.nsf/(?:[0-9a-f]{32}/|-/)?([0-9a-f]{32})", re.IGNORECASE)
_DATE_LIKE_RE = re.compile(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{4}$")
# "429/09.9TTLSB.L1.S1" -> ("429", "09"); "0615/04" -> ("615", "04"); "048318" -> ("48318", None)
_HEAD_RE = re.compile(r"^0*(\d+)[/-](\d{2}|\d{4})(?!\d)")
_BARE_RE = re.compile(r"^0*(\d{4,})$")
# Courts often type the letter O for the check digit 0 ("577/07.OTBALM").
_O_DIGIT_RE = re.compile(r"(?<=\d/\d\d\.)O")


def dgsi_unid(raw: Optional[str]) -> Optional[str]:
    """The document UNID of a dgsi.pt link written in place of the number.

    Corpus doc_ids are the Domino UNIDs, lower case.
    """
    if not raw:
        return None
    m = _UNID_RE.search(raw)
    return m.group(1).lower() if m else None


def process_head(key: Optional[str]) -> Optional[tuple[str, Optional[str]]]:
    """Serial number and two-digit year of a normalized process key.

    Tolerates what the exact key does not: leading zeros, a missing or
    mistyped (O for 0) check digit and a missing or different suffix.
    Bare numbers (old STA/STJ style) need 4+ digits to be specific enough.
    """
    if not key:
        return None
    if _DATE_LIKE_RE.match(key):
        return None
    key = _O_DIGIT_RE.sub("0", key)
    m = _HEAD_RE.match(key)
    if m:
        return m.group(1), m.group(2)[-2:]
    m = _BARE_RE.match(key)
    return (m.group(1), None) if m else None


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def pick_by_head(candidates: list[dict], key: str) -> Optional[dict]:
    """Among documents of the cited court and date, the one whose number has
    the cited serial and year. Several with the same head (e.g. .S1 and .S2)
    are decided by the longest shared prefix with the cited key; a tie
    returns None rather than a guess. Each candidate has `key`."""
    head = process_head(key)
    if not head:
        return None
    matches = [c for c in candidates if process_head(c.get("key")) == head]
    if len(matches) <= 1:
        return matches[0] if matches else None
    scored = sorted(matches, key=lambda c: _common_prefix(c["key"], key), reverse=True)
    if _common_prefix(scored[0]["key"], key) > _common_prefix(scored[1]["key"], key):
        return scored[0]
    return None


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
