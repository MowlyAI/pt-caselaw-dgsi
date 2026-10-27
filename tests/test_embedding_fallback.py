import unittest

import httpx

from api import main


def _response(status, body=None):
    request = httpx.Request("POST", f"{main.OPENROUTER_BASE}/embeddings")
    return httpx.Response(status, json=body or {}, request=request)


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    async def post(self, url, json, **kwargs):
        self.payloads.append(json)
        return self.replies.pop(0)


class EmbeddingFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.originals = (main.http_client, main.OPENROUTER_API_KEY, main.asyncio.sleep, main.EMBEDDING_MAX_RETRIES)

        async def fake_sleep(seconds):
            pass

        main.OPENROUTER_API_KEY = "test-key"
        main.asyncio.sleep = fake_sleep
        main.EMBEDDING_MAX_RETRIES = 0

    async def asyncTearDown(self):
        main.http_client, main.OPENROUTER_API_KEY, main.asyncio.sleep, main.EMBEDDING_MAX_RETRIES = self.originals

    async def test_depleted_credits_retry_on_vertex(self):
        main.http_client = FakeClient([
            _response(402, {"error": {"message": "prepayment credits are depleted"}}),
            _response(200, {"data": [{"index": 0, "embedding": [0.5]}]}),
        ])
        self.assertEqual(await main.embed_texts(["q"]), [[0.5]])
        self.assertNotIn("provider", main.http_client.payloads[0])
        self.assertEqual(main.http_client.payloads[1]["provider"], {"only": ["google-vertex"], "allow_fallbacks": False})

    async def test_request_errors_do_not_fall_back(self):
        main.http_client = FakeClient([_response(400, {"error": "bad input"})])
        with self.assertRaises(main.HTTPException):
            await main.embed_texts(["q"])
        self.assertEqual(len(main.http_client.payloads), 1)

    async def test_original_error_when_vertex_fails(self):
        main.http_client = FakeClient([_response(402, {"error": "credits"}), _response(503, {"error": "down"})])
        with self.assertRaises(main.HTTPException) as ctx:
            await main.embed_texts(["q"])
        self.assertIn("402", str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main()
