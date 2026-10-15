import unittest

from api import main
from api.disability import disability_points, injury_points


class InjuryPointsTests(unittest.TestCase):
    def test_reads_points_and_percentages(self):
        cases = {
            "19 pontos de défice funcional permanente": 19,
            "Défice Funcional Permanente da Integridade Físico-Psíquica de 16 pontos": 16,
            "IPP de 17,55%": 17.55,
            "40.52 pontos": 40.52,
            "7/100": 7,
            "Défice Funcional Permanente de 5 pontos em 100": 5,
            "6 pontos percentuais (IPP)": 6,
            "10 pontos (escala de 100)": 10,
            "0,20 (incapacidade permanente parcial)": 20,
            "40% (laboral) / 7 pontos (civil)": 40,
            "15% de incapacidade permanente parcial, com incapacidade permanente "
            "absoluta para a profissão habitual": 15,
        }
        for degree, expected in cases.items():
            with self.subTest(degree=degree):
                self.assertAlmostEqual(injury_points(degree), expected)

    def test_skips_other_scales_and_heads_of_damage(self):
        for degree, name in [
            ("Grau 4/7", "Quantum doloris"),
            ("Grau 3", "Dano estético permanente"),
            ("Grau 4 numa escala de 7", "Cicatriz abdominal"),
            ("2/7", "Repercussão nas actividades desportivas"),
            ("30%", "Incapacidade Temporária Parcial"),
            ("100% para a profissão habitual", "Perda de falanges"),
            ("0%", "Traumatismo"),
            ("Incapacidade total", "Sequelas neurológicas"),
            ("Não apurado", "Sequelas cervicais"),
        ]:
            with self.subTest(degree=degree):
                self.assertIsNone(injury_points(degree, name))

    def test_takes_the_deficit_segment_next_to_an_aesthetic_grade(self):
        self.assertEqual(injury_points(
            "Défice funcional permanente de 2 pontos; dano estético permanente de 4/7."
        ), 2)

    def test_decision_value_is_the_highest_injury_degree(self):
        metadata = {"injuries": [
            {"name": "Fratura", "disability_degree": "5 pontos"},
            {"name": "Défice funcional permanente", "disability_degree": "22 pontos"},
            {"name": "Quantum doloris", "disability_degree": "Grau 5/7"},
            "not an injury",
        ]}
        self.assertEqual(disability_points(metadata), 22)
        self.assertIsNone(disability_points({"injuries": []}))
        self.assertIsNone(disability_points(None))


class DisabilityFilterTests(unittest.TestCase):
    def test_bounds_become_a_semi_join(self):
        sql, params = main._build_filters(
            main.Filters(court=["STJ"], disability_points_gt=15, disability_points_lte=30),
            start_idx=2, alias="d",
        )
        self.assertIn(
            "d.doc_id IN (SELECT doc_id FROM document_disability_points "
            "WHERE points > $3 AND points <= $4)",
            sql,
        )
        self.assertEqual(params, [["STJ"], 15, 30])

    def test_no_bounds_leave_the_query_unchanged(self):
        sql, params = main._build_filters(main.Filters(court=["STJ"]), start_idx=1)
        self.assertNotIn("document_disability_points", sql)
        self.assertEqual(params, [["STJ"]])

    def test_request_shorthands_keep_the_disability_filter(self):
        req = main.SearchRequest(
            q_semantic="dano biológico",
            courts=["TRP"],
            filters=main.Filters(disability_points_gt=15),
        )
        resolved = req.resolved_filters()
        self.assertEqual(resolved.court, ["TRP"])
        self.assertEqual(resolved.disability_points_gt, 15)
        self.assertIsNone(main.SearchRequest(q_semantic="x").resolved_filters())


if __name__ == "__main__":
    unittest.main()
