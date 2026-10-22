"""PGR pareceres (pgrp.nsf) are scraped like the court databases."""
from scraper.config import DATABASES
from scraper.scraper import parse_document

PGR = next(db for db in DATABASES if db["short"] == "PGR")

SAMPLE = """<html><body><table>
<tr><td>Pareceres PGR</td><td>Parecer do Conselho Consultivo da PGR</td></tr>
<tr><td>Parecer:</td><td>P000062018</td></tr>
<tr><td>Relator:</td><td>MANUELA FLORES</td></tr>
<tr><td>Data da Votação:</td><td>05/18/2018</td></tr>
<tr><td>Descritores:</td><td>SISTEMA NACIONAL DE GÁS NATURAL TARIFA SOCIAL</td></tr>
<tr><td>Conclusões:</td><td>1.ª Na senda das orientações europeias presentes no Decreto-Lei n.º 77/2011,
de 20 de junho, foi estabelecida a tarifa social de gás natural.</td></tr>
</table></body></html>"""


def test_pgr_database_is_configured():
    assert PGR["db"] == "pgrp.nsf"
    assert len(PGR["view_id"]) == 32
    assert len({db["short"] for db in DATABASES}) == len(DATABASES)


def test_parecer_page_parses_with_pgr_court():
    url = "https://www.dgsi.pt/pgrp.nsf/7fc0bd52c6f5cd5a802568c0003fb410/75c800dfbd5f83cf8025824700581ac8?OpenDocument"
    doc = parse_document(SAMPLE, url, PGR)
    assert doc is not None
    assert doc["doc_id"] == "75c800dfbd5f83cf8025824700581ac8"
    assert doc["court_short"] == "PGR"
    assert "Data da Votação:" in doc["full_text"]
