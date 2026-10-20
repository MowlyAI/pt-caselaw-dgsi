import unittest

import httpx

from api import main


def _response(status, body=None, headers=None):
    request = httpx.Request("POST", f"{main.OPENROUTER_BASE}/rerank")
    return httpx.Response(status, json=body or {}, headers=headers, request=request)


def _ranked(order):
    return {"results": [{"index": i, "relevance_score": 1.0} for i in order]}


class FakeClient:
    def __init__(self, replies):
        # replies: model -> list of responses/exceptions served in order
        self.replies = {model: list(items) for model, items in replies.items()}
        self.calls = []

    async def post(self, url, json, **kwargs):
        self.calls.append(json["model"])
        item = self.replies[json["model"]].pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RerankFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.originals = (main.http_client, main.OPENROUTER_API_KEY, main.asyncio.sleep)
        self.sleeps = []

        async def fake_sleep(seconds):
            self.sleeps.append(seconds)

        main.OPENROUTER_API_KEY = "test-key"
        main.asyncio.sleep = fake_sleep
        main._rerank_cooldown_until.clear()
        self.primary, self.fallback = main.RERANK_MODELS
        self.merged = [("a", 0.3), ("b", 0.2), ("c", 0.1)]
        self.docs = {d: {"doc_id": d} for d, _ in self.merged}

    async def asyncTearDown(self):
        main.http_client, main.OPENROUTER_API_KEY, main.asyncio.sleep = self.originals
        main._rerank_cooldown_until.clear()

    async def _rerank(self):
        return await main._rerank_candidates(
            "pergunta", None, self.merged, self.docs, 60, question_only=True
        )

    async def test_primary_model_is_used_when_available(self):
        main.http_client = FakeClient({self.primary: [_response(200, _ranked([2, 0, 1]))]})
        result = await self._rerank()
        self.assertEqual([d for d, _ in result], ["c", "a", "b"])
        self.assertEqual(main.http_client.calls, [self.primary])

    async def test_429_retries_with_backoff_then_falls_back(self):
        main.http_client = FakeClient({
            self.primary: [_response(429)] * 3,
            self.fallback: [_response(200, _ranked([1, 2, 0]))],
        })
        result = await self._rerank()
        self.assertEqual([d for d, _ in result], ["b", "c", "a"])
        self.assertEqual(main.http_client.calls, [self.primary] * 3 + [self.fallback])
        self.assertEqual(self.sleeps, [0.5, 1.0])

    async def test_retry_after_header_is_respected_and_capped(self):
        main.http_client = FakeClient({
            self.primary: [
                _response(429, headers={"Retry-After": "1.5"}),
                _response(503, headers={"Retry-After": "30"}),
                _response(200, _ranked([0, 1, 2])),
            ],
        })
        await self._rerank()
        self.assertEqual(self.sleeps, [1.5, main.RERANK_MAX_RETRY_DELAY_SECONDS])

    async def test_rate_limited_model_is_skipped_during_cooldown(self):
        main.http_client = FakeClient({
            self.primary: [_response(429)] * 3,
            self.fallback: [_response(200, _ranked([0, 1, 2]))] * 2,
        })
        await self._rerank()
        await self._rerank()
        self.assertEqual(
            main.http_client.calls, [self.primary] * 3 + [self.fallback] * 2
        )

    async def test_cooldown_expires(self):
        main._rerank_cooldown_until[self.primary] = main.time.monotonic() - 1
        main.http_client = FakeClient({self.primary: [_response(200, _ranked([0, 1, 2]))]})
        await self._rerank()
        self.assertEqual(main.http_client.calls, [self.primary])

    async def test_non_retryable_error_falls_back_without_cooldown(self):
        main.http_client = FakeClient({
            self.primary: [_response(400, {"error": "bad"})],
            self.fallback: [_response(200, _ranked([0, 1, 2]))],
        })
        await self._rerank()
        self.assertEqual(main.http_client.calls, [self.primary, self.fallback])
        self.assertNotIn(self.primary, main._rerank_cooldown_until)

    async def test_network_errors_retry_then_fall_back(self):
        error = httpx.ConnectError("boom")
        main.http_client = FakeClient({
            self.primary: [error] * 3,
            self.fallback: [_response(200, _ranked([0, 1, 2]))],
        })
        await self._rerank()
        self.assertEqual(main.http_client.calls, [self.primary] * 3 + [self.fallback])

    async def test_all_models_failing_keeps_retrieval_order(self):
        main.http_client = FakeClient({
            self.primary: [_response(429)] * 3,
            self.fallback: [_response(500)] * 3,
        })
        result = await self._rerank()
        self.assertEqual(result, self.merged)

    async def test_all_models_cooling_down_makes_no_request(self):
        for model in main.RERANK_MODELS:
            main._rerank_cooldown_until[model] = main.time.monotonic() + 60
        main.http_client = FakeClient({})
        result = await self._rerank()
        self.assertEqual(result, self.merged)
        self.assertEqual(main.http_client.calls, [])

    async def test_partial_results_keep_retrieval_order(self):
        main.http_client = FakeClient({self.primary: [_response(200, _ranked([1]))]})
        result = await self._rerank()
        self.assertEqual(result, self.merged)


if __name__ == "__main__":
    unittest.main()
