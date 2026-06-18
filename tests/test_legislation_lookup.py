import asyncio
import unittest

from fastapi import HTTPException

from api import main


class LegislationLookupSqlTests(unittest.TestCase):
    def test_lookup_query_uses_normalized_table_and_two_stage_hydration(self):
        sql, params = main._build_legislation_lookup_query(
            [("Código Civil", "483.º")], "any", None, 20, 0
        )

        self.assertIn("document_legislation_citations", sql)
        self.assertIn("ranked_doc_ids", sql)
        self.assertIn("c.article <> ''", sql)
        self.assertIn("c.article LIKE $2", sql)
        self.assertIn("LIMIT $3 OFFSET $4", sql)
        self.assertEqual(params, ["Código Civil", "483.º%", 20, 0])
        self.assertNotIn("JOIN documents d ON d.doc_id = m.doc_id", sql)

    def test_lookup_query_all_mode_counts_distinct_match_keys(self):
        sql, params = main._build_legislation_lookup_query(
            [("Código Civil", "483.º"), ("Código Civil", "496.º")],
            "all",
            None,
            10,
            5,
        )

        self.assertIn("HAVING COUNT(DISTINCT match_key) = 2", sql)
        self.assertIn("UNION ALL", sql)
        self.assertEqual(
            params,
            ["Código Civil", "483.º%", "Código Civil", "496.º%", 10, 5],
        )

    def test_lookup_query_law_only_does_not_filter_article(self):
        sql, params = main._build_legislation_lookup_query(
            [("Constituição da República Portuguesa", None)], "any", None, 5, 0
        )

        self.assertIn("c.law = $1", sql)
        self.assertNotIn("c.article LIKE", sql)
        self.assertEqual(params, ["Constituição da República Portuguesa", 5, 0])

    def test_lookup_query_qualifies_document_filters(self):
        filters = main.Filters(court=["STJ"], is_auj=True)
        sql, params = main._build_legislation_lookup_query(
            [("Código do Trabalho", "394.º")], "any", filters, 20, 0
        )

        self.assertIn("c.court_short = ANY($3::text[])", sql)
        self.assertIn("c.is_auj = $4", sql)
        self.assertEqual(params, ["Código do Trabalho", "394.º%", ["STJ"], True, 20, 0])

    def test_lookup_query_uses_document_join_only_for_json_filters(self):
        filters = main.Filters(decision_type=["Acórdão"])
        sql, params = main._build_legislation_lookup_query(
            [("Código Civil", "483.º")], "any", filters, 20, 0
        )

        self.assertIn("JOIN documents d ON d.doc_id = m.doc_id", sql)
        self.assertIn("d.metadata->>'decision_type' = ANY($3::text[])", sql)
        self.assertEqual(params, ["Código Civil", "483.º%", ["Acórdão"], 20, 0])

    def test_legacy_query_preserves_jsonb_fallback(self):
        sql, params = main._build_legislation_legacy_query(
            [("Código Civil", "483.º")], "any", None, 20, 0
        )

        self.assertIn("metadata @> jsonb_build_object", sql)
        self.assertIn("jsonb_array_elements", sql)
        self.assertEqual(params, ["Código Civil", "Código Civil", "483.º%", 20, 0])


class HybridTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_hybrid_timeout_returns_504(self):
        original_search_vectors = main._search_vectors
        original_search_fts = main._search_fts
        try:
            async def timeout_vectors(*args, **kwargs):
                raise asyncio.TimeoutError()

            async def no_fts(*args, **kwargs):
                return []

            main._search_vectors = timeout_vectors
            main._search_fts = no_fts

            req = main.SearchRequest(
                q_semantic="responsabilidade civil",
                weights=main.SearchWeights(fts=0),
            )
            with self.assertRaises(HTTPException) as ctx:
                await main.search_hybrid(req)
            self.assertEqual(ctx.exception.status_code, 504)
        finally:
            main._search_vectors = original_search_vectors
            main._search_fts = original_search_fts


if __name__ == "__main__":
    unittest.main()