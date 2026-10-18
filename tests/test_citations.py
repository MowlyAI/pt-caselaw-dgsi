import unittest
from datetime import date

from api.citations import PROCESS_KEY_SQL, court_code, pick_candidate, process_key


def _doc(doc_id, court, day):
    return {"doc_id": doc_id, "court_short": court, "decision_date": day}


class ProcessKeyTests(unittest.TestCase):
    def test_normalizes_spacing_case_and_prefixes(self):
        cases = {
            "157/12.8 TUGMR.G1.S1": "157/12.8TUGMR.G1.S1",
            "6872/18.5t8vng-b.p1": "6872/18.5T8VNG-B.P1",
            "Proc. 908/12.0TYVNG-A.P1": "908/12.0TYVNG-A.P1",
            "Processo n.º 1234/19.5T8LSB.L1.S1": "1234/19.5T8LSB.L1.S1",
            "06S4278": "06S4278",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(process_key(raw), expected)

    def test_rejects_values_that_are_not_process_numbers(self):
        for raw in (None, "", "ASSENTO", "N.º", "7/2"):
            with self.subTest(raw=raw):
                self.assertIsNone(process_key(raw))

    def test_sql_expression_matches_migration(self):
        with open("migrations/20261018_documents_process_number_key_index.sql") as f:
            self.assertIn(PROCESS_KEY_SQL, f.read())


class CourtCodeTests(unittest.TestCase):
    def test_maps_abbreviations_and_names(self):
        self.assertEqual(court_code("trp", None), "TRP")
        self.assertEqual(court_code("TCA Sul", None), "TCAS")
        self.assertEqual(court_code(None, "Tribunal da Relação de Guimarães"), "TRG")
        self.assertEqual(court_code("BGH", "Supremo Tribunal de Justiça"), "STJ")
        self.assertIsNone(court_code("CJEU", "Court of Justice"))


class PickCandidateTests(unittest.TestCase):
    def test_single_candidate_consistent_with_citation(self):
        doc = _doc("a", "TRP", date(2020, 4, 30))
        self.assertEqual(pick_candidate([doc], "TRP", "2020-04-30"), doc)
        self.assertEqual(pick_candidate([doc], None, None), doc)
        # A different date alone (e.g. publication vs decision) is tolerated.
        self.assertEqual(pick_candidate([doc], "TRP", "2020-05-02"), doc)

    def test_rejects_candidate_contradicting_court_and_date(self):
        doc = _doc("a", "STJ", date(2001, 1, 1))
        self.assertIsNone(pick_candidate([doc], "TRP", "2020-04-30"))

    def test_disambiguates_by_court_then_date(self):
        trp = _doc("trp", "TRP", date(2019, 3, 1))
        stj = _doc("stj", "STJ", date(2019, 3, 1))
        self.assertEqual(pick_candidate([trp, stj], "STJ", None), stj)
        older = _doc("old", "STJ", date(2010, 1, 1))
        self.assertEqual(pick_candidate([stj, older], "STJ", "2010-01-01"), older)

    def test_ambiguous_returns_none(self):
        a = _doc("a", "TRP", date(2019, 3, 1))
        b = _doc("b", "TRL", date(2018, 3, 1))
        self.assertIsNone(pick_candidate([a, b], None, None))



class ResolveEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_one_result_per_citation_in_order(self):
        from api import main

        rows = [{"doc_id": "d1", "url": "u1", "court_short": "TRP",
                 "process_number": "6872/18.5T8VNG-B.P1",
                 "decision_date": date(2020, 4, 30), "key": "6872/18.5T8VNG-B.P1"}]
        calls = []

        class Conn:
            async def fetch(self, sql, keys):
                calls.append((sql, keys))
                return rows

        class Acquire:
            async def __aenter__(self):
                return Conn()

            async def __aexit__(self, *exc):
                return False

        class Pool:
            def acquire(self):
                return Acquire()

        original = main.db_pool
        main.db_pool = Pool()
        try:
            resp = await main.resolve_citations(main.ResolveCitationsRequest(citations=[
                main.CitedDecision(process_number="ASSENTO STJ"),
                main.CitedDecision(process_number="6872/18.5T8VNG-B.P1",
                                   court_abbreviation="TRP", decision_date="2020-04-30"),
            ]))
        finally:
            main.db_pool = original
        self.assertEqual([r.doc_id for r in resp.results], [None, "d1"])
        self.assertEqual(resp.results[1].index, 1)
        self.assertEqual(calls[0][1], ["6872/18.5T8VNG-B.P1"])
        self.assertIn("duplicate_of IS NULL", calls[0][0])


if __name__ == "__main__":
    unittest.main()
