"""Tests for the AUJ (Acórdão de Uniformização de Jurisprudência) detector.

Run with either pytest or as a plain script:
    PYTHONPATH=. python tests/test_auj_detector.py
    PYTHONPATH=. pytest tests/test_auj_detector.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractor.extractor import _is_jurisprudence_unification  # noqa: E402


# Each entry: (procedural_type, case_type, expected_flag, comment)
POSITIVE_CASES: list[tuple[str | None, str | None, str]] = [
    ("RECURSO DE FIXAÇÃO DE JURISPRUDÊNCIA", None,
     "canonical procedural_type"),
    ("RECURSO DE FIXAÇÃO DE JURISPRUDÊNCIA (PENAL)", None,
     "penal variant"),
    ("RECURSO UNIFORMIZAÇÃO DE JURISPRUDÊNCIA (CÍVEL)", None,
     "cível uniformização variant"),
    ("RECURSO UNIFORIZAÇÃO DE JURISPRUDÊNCIA", None,
     "DGSI typo: missing 'm' in uniformização"),
    (None, "Recurso de Fixação de Jurisprudência (Penal)",
     "case_type carries the signal"),
    (None, "Recurso para Fixação de Jurisprudência",
     "case_type 'para fixação'"),
    (None, "Recurso para Uniformização de Jurisprudência (Cível)",
     "case_type 'para uniformização'"),
    (None, "Recurso Extraordinário de Fixação de Jurisprudência (Penal)",
     "case_type 'extraordinário'"),
    ("RECURSO PENAL", "Recurso para Fixação de Jurisprudência",
     "generic procedural_type but AUJ case_type"),
    ("Acórdão Uniformizador", None,
     "uniformizador adjective"),
    ("  recurso para fixar jurisprudência penal  ", None,
     "lowercase + whitespace + 'fixar jurisprudência' variant"),
]

NEGATIVE_CASES: list[tuple[str | None, str | None, str]] = [
    ("REVISTA", "Revista", "ordinary revista"),
    ("RECURSO PENAL", "Recurso Penal", "ordinary penal appeal"),
    ("HABEAS CORPUS", "Habeas Corpus", "habeas corpus"),
    ("RECURSO DE REVISÃO", "Recurso de Revisão",
     "revisão is review, not uniformização"),
    ("RECLAMAÇÃO", "Reclamação para a Conferência",
     "reclamação"),
    (None, None, "both null"),
    ("", "", "both empty strings"),
    ("REVISTA EXCEPCIONAL", "Revista excepcional",
     "excepcional revista is not an AUJ"),
    ("CONFLITO DE COMPETÊNCIA", "Conflito de Competência",
     "conflito de competência"),
    ("ESCUSA/RECUSA", "Incidente de Recusa",
     "incidentes"),
]


def test_positive_cases() -> None:
    for pt, ct, comment in POSITIVE_CASES:
        assert _is_jurisprudence_unification(pt, ct) is True, \
            f"expected True for ({pt!r}, {ct!r}) — {comment}"


def test_negative_cases() -> None:
    for pt, ct, comment in NEGATIVE_CASES:
        assert _is_jurisprudence_unification(pt, ct) is False, \
            f"expected False for ({pt!r}, {ct!r}) — {comment}"


def test_either_field_sufficient() -> None:
    # Covered by positives but made explicit: signal in procedural_type only
    assert _is_jurisprudence_unification(
        "Recurso de Fixação de Jurisprudência", None
    ) is True
    # Signal in case_type only
    assert _is_jurisprudence_unification(
        None, "Recurso para Uniformização de Jurisprudência"
    ) is True


def _run_all() -> None:
    test_positive_cases()
    test_negative_cases()
    test_either_field_sufficient()
    total = len(POSITIVE_CASES) + len(NEGATIVE_CASES) + 2
    print(f"OK — {total} assertions passed.")


if __name__ == "__main__":
    _run_all()
