"""Offline SDK regressions; run with `python -m unittest discover -s tests`."""

import asyncio
import importlib.util
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, patch

from openai import AsyncOpenAI


class AsyncChatTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Load the real provider without initializing optional HF/Google providers.
        path = Path(__file__).resolve().parents[1] / "llms/providers/openai_utils.py"
        spec = importlib.util.spec_from_file_location("openai_utils_under_test", path)
        cls.provider = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}):
            os.environ.pop("OPENAI_API_BASE", None)
            spec.loader.exec_module(cls.provider)

    @classmethod
    def tearDownClass(cls):
        cls.provider.client.close()
        asyncio.run(cls.provider.aclient.close())

    async def generate(self, prompts, statuses=None):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                prompt = body["messages"][0]["content"]
                status = (statuses or {}).get(prompt, 200)
                if status == 200:
                    response = {
                        "id": "chatcmpl-offline",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": None
                                    if prompt == "null-content"
                                    else prompt,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                else:
                    response = {
                        "error": {"message": "offline failure", "type": "test_error"}
                    }
                encoded = json.dumps(response).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        thread.start()
        try:
            async with AsyncOpenAI(
                api_key="offline-test",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                max_retries=0,
                timeout=5,
            ) as client:
                with (
                    patch.object(self.provider, "aclient", client),
                    patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}),
                ):
                    result = await self.provider.agenerate_from_openai_chat_completion(
                        [[{"role": "user", "content": prompt}] for prompt in prompts],
                        engine="offline-model",
                        temperature=0.25,
                        max_tokens=64,
                        top_p=0.9,
                        context_length=1024,
                    )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)
        return result, requests

    async def test_sdk_chat_completions_preserve_batch_order(self):
        result, requests = await self.generate(["first", "second"])
        self.assertEqual(result, ["first", "second"])
        self.assertEqual(len(requests), 2)
        for request in requests:
            self.assertEqual(request["model"], "offline-model")
            self.assertEqual(request["max_tokens"], 64)
            self.assertEqual(request["temperature"], 0.25)
            self.assertEqual(request["top_p"], 0.9)

    async def test_unicode_and_empty_completions(self):
        result, _ = await self.generate(["中文 🐋", ""])
        self.assertEqual(result, ["中文 🐋", ""])

    async def test_null_content_is_an_empty_string(self):
        result, _ = await self.generate(["null-content"])
        self.assertEqual(result, [""])

    async def test_api_error_keeps_empty_string_fallback(self):
        result, requests = await self.generate(["failed"], {"failed": 500})
        self.assertEqual(result, [""])
        self.assertEqual(len(requests), 1)

    async def test_exhausted_rate_limit_keeps_empty_string_fallback(self):
        with patch.object(self.provider.asyncio, "sleep", new_callable=AsyncMock):
            result, requests = await self.generate(["limited"], {"limited": 429})
        self.assertEqual(result, [""])
        self.assertEqual(len(requests), 3)

    async def test_success_and_failure_can_share_a_batch(self):
        result, _ = await self.generate(["first", "failed", "last"], {"failed": 500})
        self.assertEqual(result, ["first", "", "last"])

    async def test_empty_batch(self):
        result, requests = await self.generate([])
        self.assertEqual(result, [])
        self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
