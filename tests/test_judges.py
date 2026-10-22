import asyncio
import unittest
from unittest import mock

from api import main
from api.judges import judge_key, judge_name, query_tokens, relator_from_full_text


class JudgeKeyTests(unittest.TestCase):
    def test_ignores_case_accents_and_punctuation(self):
        self.assertEqual(judge_key("José Correia"), "JOSE CORREIA")
        self.assertEqual(judge_key("JOSE  CORREIA"), "JOSE CORREIA")
        self.assertEqual(judge_key("E. Sequeira"), "E SEQUEIRA")
        self.assertEqual(judge_key("Tânia Meireles da Cunha"), "TANIA MEIRELES DA CUNHA")
        self.assertEqual(judge_key(None), "")

    def test_query_tokens_are_normalised_words(self):
        self.assertEqual(query_tokens("  j. corrêia "), ["J", "CORREIA"])
        self.assertEqual(query_tokens("%_"), [])


class RelatorTests(unittest.TestCase):
    PAGE = (
        "Acórdão do Tribunal Central Administrativo Sul\nProcesso:\n123/20.1BELSB\n"
        "Secção:\nCT\nData do Acordão:\n05/03/2024\nRelator:\nTÂNIA MEIRELES DA CUNHA\n"
        "Descritores:\nIRC\n"
    )

    def test_reads_the_page_field(self):
        self.assertEqual(relator_from_full_text(self.PAGE), "TÂNIA MEIRELES DA CUNHA")
        self.assertEqual(relator_from_full_text("Relator: Maria  Cardoso\nDescritores:"), "Maria Cardoso")

    def test_rejects_labels_and_missing_fields(self):
        self.assertIsNone(relator_from_full_text("Relator:\nDescritores:\nIRC"))
        self.assertIsNone(relator_from_full_text("Processo:\n1/20"))
        self.assertIsNone(relator_from_full_text(None))

    def test_falls_back_to_the_extracted_name(self):
        self.assertEqual(judge_name(self.PAGE, {"judge_name": "OUTRO"}), "TÂNIA MEIRELES DA CUNHA")
        self.assertEqual(judge_name(None, {"judge_name": "GARCIA CALEJO"}), "GARCIA CALEJO")
        self.assertIsNone(judge_name("sem relator", {"judge_name": None}))


class JudgeFilterTests(unittest.TestCase):
    def test_names_become_keys_in_a_semi_join(self):
        filters = main.Filters(court=["STJ"], judge=["José Correia", "JOSE CORREIA", "E. Sequeira"])
        self.assertEqual(filters.judge, ["JOSE CORREIA", "E SEQUEIRA"])
        sql, params = main._build_filters(filters, start_idx=2, alias="d")
        self.assertIn(
            "d.doc_id IN (SELECT doc_id FROM document_judges WHERE judge_key = ANY($3::text[]))",
            sql,
        )
        self.assertEqual(params, [["STJ"], ["JOSE CORREIA", "E SEQUEIRA"]])

    def test_blank_names_mean_no_filter(self):
        self.assertIsNone(main.Filters(judge=[" ", "."]).judge)
        sql, params = main._build_filters(main.Filters(court=["STJ"]), start_idx=1)
        self.assertNotIn("document_judges", sql)
        self.assertEqual(params, [["STJ"]])

    def test_document_only_filters_carry_the_judge(self):
        sql, params, idx = main._build_document_json_filters(main.Filters(judge=["Garcia Calejo"]), 4)
        self.assertIn("judge_key = ANY($4::text[])", sql)
        self.assertEqual((params, idx), ([["GARCIA CALEJO"]], 5))

    def test_vector_query_ranks_a_judge_exactly(self):
        captured = []

        class Conn:
            async def fetch(self, sql, *args):
                captured.append(sql)
                return []

        class Acquire:
            async def __aenter__(self):
                return Conn()

            async def __aexit__(self, *exc):
                return False

        pool = mock.Mock()
        pool.acquire.return_value = Acquire()
        with mock.patch.object(main, "db_pool", pool):
            asyncio.run(main._vector_query("[0]", "embedding", 10, main.Filters(judge=["X Y"])))
            asyncio.run(main._vector_query("[0]", "embedding", 10, main.Filters(court=["STJ"])))
        self.assertIn("ORDER BY (embedding <=> $1::halfvec) + 0", captured[0])
        self.assertIn("ORDER BY (embedding <=> $1::halfvec) LIMIT", captured[1])


class BrowseTests(unittest.TestCase):
    def test_requires_a_judge(self):
        with self.assertRaises(main.HTTPException) as ctx:
            asyncio.run(main.browse(main.BrowseRequest(filters=main.Filters(court=["STJ"]))))
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
