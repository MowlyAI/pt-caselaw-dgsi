import unittest

import httpx

from api import main

VOYAGE = "voyage/rerank-2.5"
OR_VOYAGE = "openrouter/voyageai/rerank-2.5"
OR_COHERE = "openrouter/cohere/rerank-v3.5"


def _response(status, body=None, headers=None):
    request = httpx.Request("POST", "https://example.test/rerank")
    return httpx.Response(status, json=body or {}, headers=headers, request=request)


def _ranked(order):
    return {"results": [{"index": i, "relevance_score": 1.0} for i in order]}


def _voyage_ranked(order):
    # Voyage's own API answers `data`, with scores instead of positions.
    n = len(order)
    return {
        "object": "list",
        "data": [{"index": i, "relevance_score": (n - pos) / n} for pos, i in enumerate(order)],
        "model": "rerank-2.5",
        "usage": {"total_tokens": 10},
    }


class FakeClient:
    def __init__(self, replies):
        # replies: route name -> list of responses/exceptions served in order
        self.replies = {name: list(items) for name, items in replies.items()}
        self.calls = []
        self.bodies = []

    async def post(self, url, json, headers=None, **kwargs):
        name = VOYAGE if url == main.VOYAGE_RERANK_URL else f"openrouter/{json['model']}"
        self.calls.append(name)
        self.bodies.append((name, json, headers))
        item = self.replies[name].pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RerankRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.originals = (
            main.http_client, main.OPENROUTER_API_KEY, main.VOYAGE_API_KEY, main.asyncio.sleep,
        )
        self.sleeps = []

        async def fake_sleep(seconds):
            self.sleeps.append(seconds)

        main.OPENROUTER_API_KEY = "test-openrouter-key"
        main.VOYAGE_API_KEY = "test-voyage-key"
        main.asyncio.sleep = fake_sleep
        main._rerank_cooldown_until.clear()
        main._rerank_route_status.clear()

    async def asyncTearDown(self):
        (main.http_client, main.OPENROUTER_API_KEY, main.VOYAGE_API_KEY,
         main.asyncio.sleep) = self.originals
        main._rerank_cooldown_until.clear()
        main._rerank_route_status.clear()

    def use(self, replies):
        main.http_client = FakeClient(replies)
        return main.http_client

    async def test_voyage_direct_is_first_and_normalized(self):
        client = self.use({VOYAGE: [_response(200, _voyage_ranked([1, 0]))]})
        result = await main._post_rerank("q", ["a", "b"])
        self.assertEqual(result[0], VOYAGE)
        self.assertEqual([item["index"] for item in result[1]], [1, 0])
        name, body, headers = client.bodies[0]
        self.assertEqual(body["model"], "rerank-2.5")
        self.assertEqual(body["top_k"], 2)
        self.assertNotIn("top_n", body)
        self.assertEqual(headers["Authorization"], "Bearer test-voyage-key")

    async def test_voyage_unsorted_data_is_sorted_by_score(self):
        body = {"data": [
            {"index": 0, "relevance_score": 0.1},
            {"index": 2, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5},
        ]}
        self.use({VOYAGE: [_response(200, body)]})
        _, results = await main._post_rerank("q", ["a", "b", "c"])
        self.assertEqual([item["index"] for item in results], [2, 1, 0])

    async def test_without_voyage_key_openrouter_voyage_is_first(self):
        main.VOYAGE_API_KEY = None
        client = self.use({OR_VOYAGE: [_response(200, _ranked([1, 0]))]})
        result = await main._post_rerank("q", ["a", "b"])
        self.assertEqual(result, (OR_VOYAGE, _ranked([1, 0])["results"]))
        self.assertEqual(client.bodies[0][1]["top_n"], 2)

    async def test_429_moves_on_at_once_without_retry_or_sleep(self):
        client = self.use({
            VOYAGE: [_response(429)],
            OR_VOYAGE: [_response(200, _ranked([0, 1]))],
        })
        result = await main._post_rerank("q", ["a", "b"])
        self.assertEqual(result[0], OR_VOYAGE)
        self.assertEqual(client.calls, [VOYAGE, OR_VOYAGE])
        self.assertEqual(self.sleeps, [])
        self.assertIn(VOYAGE, main._rerank_cooldown_until)

    async def test_route_cooling_down_is_skipped(self):
        main._rerank_cooldown_until[VOYAGE] = main.time.monotonic() + 60
        client = self.use({OR_VOYAGE: [_response(200, _ranked([0, 1]))]})
        await main._post_rerank("q", ["a", "b"])
        self.assertEqual(client.calls, [OR_VOYAGE])

    async def test_expired_cooldown_tries_route_again(self):
        main._rerank_cooldown_until[VOYAGE] = main.time.monotonic() - 1
        client = self.use({VOYAGE: [_response(200, _voyage_ranked([0, 1]))]})
        result = await main._post_rerank("q", ["a", "b"])
        self.assertEqual(result[0], VOYAGE)
        self.assertEqual(client.calls, [VOYAGE])

    async def test_retry_after_sets_cooldown_capped(self):
        self.use({
            VOYAGE: [_response(429, headers={"Retry-After": "99999"})],
            OR_VOYAGE: [_response(200, _ranked([0, 1]))],
        })
        before = main.time.monotonic()
        await main._post_rerank("q", ["a", "b"])
        remaining = main._rerank_cooldown_until[VOYAGE] - before
        self.assertLessEqual(remaining, main.RERANK_MAX_COOLDOWN_SECONDS + 1)
        self.assertGreater(remaining, main.RERANK_COOLDOWN_SECONDS)

    async def test_rejected_key_cools_down_longest(self):
        self.use({
            VOYAGE: [_response(403, {"detail": "bad key"})],
            OR_VOYAGE: [_response(200, _ranked([0, 1]))],
        })
        before = main.time.monotonic()
        result = await main._post_rerank("q", ["a", "b"])
        self.assertEqual(result[0], OR_VOYAGE)
        self.assertGreaterEqual(
            main._rerank_cooldown_until[VOYAGE] - before, main.RERANK_MAX_COOLDOWN_SECONDS - 1
        )

    async def test_server_error_moves_on_without_cooldown(self):
        client = self.use({
            VOYAGE: [_response(500)],
            OR_VOYAGE: [_response(200, _ranked([0, 1]))],
        })
        await main._post_rerank("q", ["a", "b"])
        self.assertEqual(client.calls, [VOYAGE, OR_VOYAGE])
        self.assertNotIn(VOYAGE, main._rerank_cooldown_until)

    async def test_network_error_moves_on_once(self):
        client = self.use({
            VOYAGE: [httpx.ConnectError("boom")],
            OR_VOYAGE: [_response(200, _ranked([0, 1]))],
        })
        result = await main._post_rerank("q", ["a", "b"])
        self.assertEqual(result[0], OR_VOYAGE)
        self.assertEqual(client.calls, [VOYAGE, OR_VOYAGE])
        self.assertEqual(self.sleeps, [])

    async def test_cohere_is_last_resort(self):
        client = self.use({
            VOYAGE: [_response(429)],
            OR_VOYAGE: [_response(429)],
            OR_COHERE: [_response(200, _ranked([1, 0]))],
        })
        result = await main._post_rerank("q", ["a", "b"])
        self.assertEqual(result[0], OR_COHERE)
        self.assertEqual(client.calls, [VOYAGE, OR_VOYAGE, OR_COHERE])

    async def test_all_failing_returns_none_with_one_call_each(self):
        client = self.use({
            VOYAGE: [_response(503)],
            OR_VOYAGE: [_response(429)],
            OR_COHERE: [_response(500)],
        })
        self.assertIsNone(await main._post_rerank("q", ["a", "b"]))
        self.assertEqual(client.calls, [VOYAGE, OR_VOYAGE, OR_COHERE])

    async def test_all_cooling_down_makes_no_request(self):
        for name in (VOYAGE, OR_VOYAGE, OR_COHERE):
            main._rerank_cooldown_until[name] = main.time.monotonic() + 60
        client = self.use({})
        self.assertIsNone(await main._post_rerank("q", ["a", "b"]))
        self.assertEqual(client.calls, [])

    async def test_health_reports_routes_without_keys(self):
        self.use({
            VOYAGE: [_response(403)],
            OR_VOYAGE: [_response(200, _ranked([0, 1]))],
        })
        await main._post_rerank("q", ["a", "b"])
        health = main.rerank_health()
        names = [route["name"] for route in health["routes"]]
        self.assertEqual(names, [VOYAGE, OR_VOYAGE, OR_COHERE])
        self.assertEqual(health["routes"][0]["last_outcome"], "HTTP 403")
        self.assertGreater(health["routes"][0]["cooling_down_seconds"], 0)
        self.assertEqual(health["routes"][1]["last_outcome"], "ok")
        self.assertNotIn("test-voyage-key", repr(health))

    async def test_rerank_candidates_reports_route_and_reorders(self):
        self.use({VOYAGE: [_response(200, _voyage_ranked([1, 0]))]})
        main._reranked_by.set(None)
        merged = [("a", 0.0101), ("b", 0.01)]
        docs = {"a": {"summary": "A"}, "b": {"summary": "B"}}
        result = await main._rerank_candidates("q", None, merged, docs, 60)
        self.assertEqual([doc_id for doc_id, _ in result], ["b", "a"])
        self.assertEqual(main._reranked_by.get(), VOYAGE)

    async def test_partial_results_keep_order_and_report_none(self):
        self.use({VOYAGE: [_response(200, _voyage_ranked([1]))]})
        main._reranked_by.set(None)
        merged = [("a", 0.0101), ("b", 0.01)]
        docs = {"a": {"summary": "A"}, "b": {"summary": "B"}}
        result = await main._rerank_candidates("q", None, merged, docs, 60)
        self.assertEqual(result, merged)
        self.assertIsNone(main._reranked_by.get())


if __name__ == "__main__":
    unittest.main()
