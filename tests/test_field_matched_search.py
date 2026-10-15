import unittest

from api import main


class FieldMatchedSearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.originals = (main._search_vectors, main._search_fts, main._fetch_docs,
                          main._rerank_candidates)
        self.vector_calls = []
        self.rerank_queries = []
        self.rerank_modes = []

        async def fake_vectors(q, fields, limit, filters):
            self.vector_calls.append((q, list(fields)))
            return {f: [(f"{f}-doc", 0.9)] for f in fields}

        async def fake_fts(q, limit, filters):
            return [("fts-doc", 0.5)]

        async def fake_docs(doc_ids):
            return {d: {"doc_id": d, "url": "u", "court_short": "STJ"} for d in doc_ids}

        async def fake_rerank(semantic_query, keyword_query, merged, *args, **kwargs):
            self.rerank_queries.append(semantic_query)
            self.rerank_modes.append(kwargs.get("question_only"))
            return merged

        main._search_vectors = fake_vectors
        main._search_fts = fake_fts
        main._fetch_docs = fake_docs
        main._rerank_candidates = fake_rerank

    async def asyncTearDown(self):
        (main._search_vectors, main._search_fts, main._fetch_docs,
         main._rerank_candidates) = self.originals

    async def test_each_column_embeds_its_own_text(self):
        req = main.SearchRequest(
            q_semantic="pergunta do utilizador",
            q_keywords=['"despedimento ilicito" OR "justa causa"'],
            q_semantic_fields=main.SemanticFieldQueries(
                embedding="frase do caso", embedding_ratio="criterio juridico"
            ),
        )
        res = await main.search_hybrid(req)
        calls = dict((q, f) for q, f in self.vector_calls)
        self.assertEqual(calls["frase do caso"], ["embedding"])
        self.assertEqual(calls["criterio juridico"], ["embedding_ratio"])
        # Columns without their own text fall back to q_semantic.
        self.assertEqual(calls["pergunta do utilizador"], ["embedding_context"])
        self.assertEqual(self.rerank_queries, ["pergunta do utilizador"])
        self.assertEqual(res.query_semantic, "pergunta do utilizador")
        self.assertEqual(
            sorted(r.doc_id for r in res.results),
            ["embedding-doc", "embedding_context-doc", "embedding_ratio-doc", "fts-doc"],
        )

    async def test_rerank_mode_reaches_the_reranker(self):
        await main.search_hybrid(main.SearchRequest(q_semantic="p", weights=main.SearchWeights(fts=0)))
        await main.search_hybrid(main.SearchRequest(
            q_semantic="p", weights=main.SearchWeights(fts=0), rerank_mode="question"))
        self.assertEqual(self.rerank_modes, [False, True])

    async def test_shared_text_is_embedded_once(self):
        req = main.SearchRequest(q_semantic="pergunta", weights=main.SearchWeights(fts=0))
        await main.search_hybrid(req)
        self.assertEqual(
            self.vector_calls,
            [("pergunta", ["embedding", "embedding_context", "embedding_ratio"])],
        )

    async def test_fields_alone_cover_vector_search(self):
        req = main.SearchRequest(
            q_semantic_fields=main.SemanticFieldQueries(embedding="frase do caso"),
            weights=main.SearchWeights(embedding_context=0, embedding_ratio=0, fts=0),
        )
        await main.search_hybrid(req)
        self.assertEqual(self.vector_calls, [("frase do caso", ["embedding"])])
        self.assertEqual(self.rerank_queries, ["frase do caso"])

    async def test_missing_text_for_a_column_is_rejected(self):
        req = main.SearchRequest(
            q_semantic_fields=main.SemanticFieldQueries(embedding="frase do caso"),
            weights=main.SearchWeights(fts=0),
        )
        with self.assertRaises(main.HTTPException) as ctx:
            await main.search_hybrid(req)
        self.assertEqual(ctx.exception.status_code, 400)


class _CapturingRerankClient:
    def __init__(self, results):
        self.results = results
        self.queries = []

    async def post(self, *args, **kwargs):
        self.queries.append(kwargs["json"]["query"])
        results = self.results

        class _Response:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                return None

            def json(self):
                return {"results": results}

        return _Response()


class QuestionRerankTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.saved = (main.http_client, main.OPENROUTER_API_KEY, main.RERANK_WEIGHT)
        main.OPENROUTER_API_KEY = "test-key"
        main.RERANK_WEIGHT = 1.0
        self.docs = {d: {"doc_id": d, "metadata": {}} for d in ("a", "b", "c")}
        # The reranker puts c first, b second, a last; the fused order is a, b, c.
        main.http_client = _CapturingRerankClient([
            {"index": 2, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5},
            {"index": 0, "relevance_score": 0.1},
        ])
        self.fused = [("a", 0.9), ("b", 0.5), ("c", 0.1)]

    async def asyncTearDown(self):
        main.http_client, main.OPENROUTER_API_KEY, main.RERANK_WEIGHT = self.saved

    async def test_question_mode_returns_reranker_order_and_reads_only_the_question(self):
        out = await main._rerank_candidates(
            "pergunta", '"a b" OR "c d"', self.fused, self.docs, 50, question_only=True
        )
        self.assertEqual([d for d, _ in out], ["c", "b", "a"])
        self.assertEqual(main.http_client.queries, ["Consulta: pergunta"])

    async def test_blend_mode_keeps_fused_scores_and_keywords(self):
        out = await main._rerank_candidates("pergunta", '"a b" OR "c d"', self.fused, self.docs, 50)
        self.assertEqual([d for d, _ in out], ["a", "b", "c"])
        self.assertIn("Termos jurídicos essenciais: a b, c d", main.http_client.queries[0])

    async def test_question_mode_without_question_falls_back_to_blend(self):
        out = await main._rerank_candidates(None, "termos", self.fused, self.docs, 50, question_only=True)
        self.assertEqual([d for d, _ in out], ["a", "b", "c"])


class PlainKeywordsTests(unittest.TestCase):
    def test_strips_websearch_syntax(self):
        self.assertEqual(
            main._plain_keywords('"despedimento ilicito" OR "justa causa"  OR  prazo'),
            "despedimento ilicito, justa causa, prazo",
        )

    def test_plain_terms_unchanged(self):
        self.assertEqual(main._plain_keywords("despedimento justa causa"),
                         "despedimento justa causa")


if __name__ == "__main__":
    unittest.main()
