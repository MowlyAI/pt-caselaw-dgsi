import unittest

from api import main


class FieldMatchedSearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.originals = (main._search_vectors, main._search_fts, main._fetch_docs,
                          main._rerank_candidates)
        self.vector_calls = []
        self.rerank_queries = []

        async def fake_vectors(q, fields, limit, filters):
            self.vector_calls.append((q, list(fields)))
            return {f: [(f"{f}-doc", 0.9)] for f in fields}

        async def fake_fts(q, limit, filters):
            return [("fts-doc", 0.5)]

        async def fake_docs(doc_ids):
            return {d: {"doc_id": d, "url": "u", "court_short": "STJ"} for d in doc_ids}

        async def fake_rerank(semantic_query, keyword_query, merged, *args, **kwargs):
            self.rerank_queries.append(semantic_query)
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
