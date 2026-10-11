import asyncio
import json
import unittest

import httpx
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
        self.assertIn("c.article = $2 OR c.article LIKE $3", sql)
        self.assertIn("LIMIT $4 OFFSET $5", sql)
        self.assertEqual(params, ["Código Civil", "483.º", "483.º,%", 20, 0])
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
            [
                "Código Civil", "483.º", "483.º,%",
                "Código Civil", "496.º", "496.º,%",
                10, 5,
            ],
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

        self.assertIn("c.court_short = ANY($4::text[])", sql)
        self.assertIn("c.is_auj = $5", sql)
        self.assertEqual(
            params,
            ["Código do Trabalho", "394.º", "394.º,%", ["STJ"], True, 20, 0],
        )

    def test_lookup_query_uses_document_join_only_for_json_filters(self):
        filters = main.Filters(decision_type=["Acórdão"])
        sql, params = main._build_legislation_lookup_query(
            [("Código Civil", "483.º")], "any", filters, 20, 0
        )

        self.assertIn("JOIN documents d ON d.doc_id = m.doc_id", sql)
        self.assertIn("d.metadata->>'decision_type' = ANY($4::text[])", sql)
        self.assertEqual(
            params,
            ["Código Civil", "483.º", "483.º,%", ["Acórdão"], 20, 0],
        )

    def test_article_match_stops_at_canonical_boundary(self):
        sql, params = main._build_legislation_lookup_query(
            [("Código do Trabalho", "12.º")], "any", None, 20, 0
        )

        self.assertIn("c.article = $2 OR c.article LIKE $3", sql)
        self.assertEqual(params[:3], ["Código do Trabalho", "12.º", "12.º,%"])

    def test_lettered_article_normalization_has_one_ordinal_marker(self):
        self.assertEqual(main._normalize_article_prefix("67-A"), "67.º-A")

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


class _FakeRerankResponse:
    def __init__(self, results, status_code=200):
        self._results = results
        self.status_code = status_code
        self.headers = {}

    def raise_for_status(self):
        return None

    def json(self):
        return {"results": self._results}


class _FakeJsonResponse:
    status_code = 200
    headers = {}

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class _FakeJsonClient:
    def __init__(self, body):
        self.body = body

    async def post(self, *args, **kwargs):
        return _FakeJsonResponse(self.body)


class _FakeRerankClient:
    def __init__(self, results=None, error=None, statuses=None):
        self.results = results
        self.error = error
        self.statuses = list(statuses or [])
        self.calls = 0

    async def post(self, *args, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        status = self.statuses.pop(0) if self.statuses else 200
        return _FakeRerankResponse(self.results, status)


class HybridRerankTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_client = main.http_client
        self.original_key = main.OPENROUTER_API_KEY
        self.original_weight = main.RERANK_WEIGHT
        self.original_guard = main.RERANK_LEXICAL_GUARD
        main.OPENROUTER_API_KEY = "test-key"
        main.RERANK_WEIGHT = 4.0
        self.docs = {
            "a": {"doc_id": "a", "summary": "Caso apenas relacionado", "metadata": {}},
            "b": {"doc_id": "b", "summary": "Caso diretamente relevante", "metadata": {}},
        }

    async def asyncTearDown(self):
        main.http_client = self.original_client
        main.OPENROUTER_API_KEY = self.original_key
        main.RERANK_WEIGHT = self.original_weight
        main.RERANK_LEXICAL_GUARD = self.original_guard

    async def test_reranker_changes_order_without_new_response_source(self):
        main.http_client = _FakeRerankClient([
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
        ])

        reranked = await main._rerank_candidates(
            "consulta", "termos", [("a", 0.05), ("b", 0.049)], self.docs, 50
        )

        self.assertEqual([doc_id for doc_id, _ in reranked], ["b", "a"])

    async def test_reranker_failure_preserves_order_and_scores(self):
        original = [("a", 0.05), ("b", 0.049)]
        main.http_client = _FakeRerankClient(error=RuntimeError("provider unavailable"))

        reranked = await main._rerank_candidates(
            "consulta", "termos", original, self.docs, 50
        )

        self.assertEqual(reranked, original)

    async def test_reranker_retries_transient_rate_limit(self):
        client = _FakeRerankClient(
            results=[
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ],
            statuses=[429, 200],
        )
        main.http_client = client

        reranked = await main._rerank_candidates(
            "consulta", "termos", [("a", 0.05), ("b", 0.049)], self.docs, 50
        )

        self.assertEqual(client.calls, 2)
        self.assertEqual([doc_id for doc_id, _ in reranked], ["b", "a"])

    def _six_candidates(self):
        docs = {
            str(index): {"doc_id": str(index), "summary": str(index), "metadata": {}}
            for index in range(1, 7)
        }
        main.http_client = _FakeRerankClient([
            {"index": 5, "relevance_score": 0.99},
            {"index": 0, "relevance_score": 0.8},
            {"index": 1, "relevance_score": 0.7},
            {"index": 2, "relevance_score": 0.6},
            {"index": 3, "relevance_score": 0.5},
            {"index": 4, "relevance_score": 0.4},
        ])
        merged = [(str(index), 0.1 - index * 0.001) for index in range(1, 7)]
        return docs, merged

    async def test_reranker_can_promote_nonlexical_candidate_by_default(self):
        docs, merged = self._six_candidates()

        reranked = await main._rerank_candidates(
            "consulta", "termos", merged, docs, 50, lexical_promotion_ids=set()
        )

        self.assertIn("6", [doc_id for doc_id, _ in reranked[:5]])

    async def test_nonlexical_candidate_cannot_displace_original_top_five(self):
        main.RERANK_LEXICAL_GUARD = True
        docs = {
            str(index): {"doc_id": str(index), "summary": str(index), "metadata": {}}
            for index in range(1, 7)
        }
        main.http_client = _FakeRerankClient([
            {"index": 5, "relevance_score": 0.99},
            {"index": 0, "relevance_score": 0.8},
            {"index": 1, "relevance_score": 0.7},
            {"index": 2, "relevance_score": 0.6},
            {"index": 3, "relevance_score": 0.5},
            {"index": 4, "relevance_score": 0.4},
        ])
        merged = [(str(index), 0.1 - index * 0.001) for index in range(1, 7)]

        reranked = await main._rerank_candidates(
            "consulta", "termos", merged, docs, 50, lexical_promotion_ids=set()
        )

        self.assertNotIn("6", [doc_id for doc_id, _ in reranked[:5]])


class PortugueseFtsTests(unittest.TestCase):
    def test_normalization_folds_accents_and_preserves_websearch_operators(self):
        query = '“responsabilidade médica” OR acção -prescrição'

        normalized = main._normalize_portuguese_fts_query(query)

        self.assertEqual(normalized, '"responsabilidade medica" OR accao -prescricao')


class LegislationIntentResolverTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolver_normalizes_structured_portuguese_citation(self):
        original_client = main.http_client
        original_key = main.OPENROUTER_API_KEY
        main.OPENROUTER_API_KEY = "test-key"
        main.http_client = _FakeJsonClient({
            "choices": [{
                "message": {
                    "content": '{"citations":[{"law":"CT","article":"12-A",'
                               '"reason":"Presunção para plataformas"}]}'
                }
            }]
        })
        try:
            resolved = await main._resolve_legislation_intent("estafetas de plataforma")
        finally:
            main.http_client = original_client
            main.OPENROUTER_API_KEY = original_key

        self.assertEqual(resolved, [{
            "law": "Código do Trabalho",
            "article": "12.º-A",
            "reason": "Presunção para plataformas",
        }])


class _FakeEmbeddingResponse:
    headers = {}

    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class _FakeEmbeddingClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def post(self, *args, **kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class EmbeddingRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_rate_limit_is_retried(self):
        original_client = main.http_client
        client = _FakeEmbeddingClient([
            _FakeEmbeddingResponse(429, {"error": "rate limited"}),
            _FakeEmbeddingResponse(200, {"data": [{"embedding": [0.1, 0.2]}]}),
        ])
        main.http_client = client
        try:
            embedding = await main.embed_query("consulta")
        finally:
            main.http_client = original_client

        self.assertEqual(embedding, [0.1, 0.2])
        self.assertEqual(client.calls, 2)

    async def test_transient_network_error_is_retried(self):
        original_client = main.http_client
        client = _FakeEmbeddingClient([
            httpx.ReadError("connection reset"),
            _FakeEmbeddingResponse(200, {"data": [{"embedding": [0.3]}]}),
        ])
        main.http_client = client
        try:
            embedding = await main.embed_query("consulta")
        finally:
            main.http_client = original_client

        self.assertEqual(embedding, [0.3])
        self.assertEqual(client.calls, 2)


if __name__ == "__main__":
    unittest.main()
