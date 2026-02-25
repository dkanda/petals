import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import json
import asyncio
import time
import threading

# Mock hivemind and other dependencies before importing petals
mock_hivemind = MagicMock()
sys.modules["hivemind"] = mock_hivemind
sys.modules["hivemind.utils"] = mock_hivemind.utils
sys.modules["hivemind.utils.logging"] = mock_hivemind.utils.logging
sys.modules["hivemind.utils.asyncio"] = MagicMock()
sys.modules["hivemind.utils.streaming"] = MagicMock()
sys.modules["hivemind.utils.mpfuture"] = MagicMock()
sys.modules["hivemind.utils.threading"] = MagicMock()
sys.modules["hivemind.utils.timed_storage"] = MagicMock()
sys.modules["hivemind.utils.nested"] = MagicMock()
sys.modules["hivemind.utils.serializer"] = MagicMock()
sys.modules["hivemind.moe"] = MagicMock()
sys.modules["hivemind.moe.client"] = MagicMock()
sys.modules["hivemind.moe.client.remote_expert_worker"] = MagicMock()
sys.modules["hivemind.moe.expert_uid"] = MagicMock()
sys.modules["hivemind.moe.server"] = MagicMock()
sys.modules["hivemind.moe.server.connection_handler"] = MagicMock()
sys.modules["hivemind.moe.server.module_backend"] = MagicMock()
sys.modules["hivemind.p2p"] = MagicMock()
sys.modules["hivemind.p2p.p2p_daemon"] = MagicMock()
sys.modules["hivemind.p2p.p2p_daemon_bindings"] = MagicMock()
sys.modules["hivemind.p2p.p2p_daemon_bindings.control"] = MagicMock()
sys.modules["hivemind.proto"] = MagicMock()
sys.modules["hivemind.utils.tensor_descr"] = MagicMock()
sys.modules["hivemind.utils.networking"] = MagicMock()
sys.modules["hivemind.compression"] = MagicMock()
sys.modules["hivemind.compression.serialization"] = MagicMock()
sys.modules["hivemind.compression.base"] = MagicMock()
sys.modules["hivemind.dht"] = MagicMock()
sys.modules["hivemind.dht.node"] = MagicMock()
sys.modules["hivemind.optim"] = MagicMock()
sys.modules["hivemind.optim.progress_tracker"] = MagicMock()


# bitsandbytes is installed
# sys.modules["bitsandbytes"] = MagicMock()

# Mock tensor_parallel as it was not installed (failed with hivemind?)
sys.modules["tensor_parallel"] = MagicMock()
sys.modules["tensor_parallel.tensor_parallel"] = MagicMock()
sys.modules["tensor_parallel.slicing_configs"] = MagicMock()
sys.modules["tensor_parallel.sliced_model"] = MagicMock()
sys.modules["tensor_parallel.utils"] = MagicMock()


# Mock humanfriendly
sys.modules["humanfriendly"] = MagicMock()

# Mock cpufeature
sys.modules["cpufeature"] = MagicMock()

# peft is installed
# sys.modules["peft"] = MagicMock()

# Mock Dijkstar
sys.modules["dijkstar"] = MagicMock()

from fastapi.testclient import TestClient
import torch

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from petals.cli.run_http_server import create_app

class TestHTTPServer(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_args = MagicMock()
        self.mock_args.model = "test-model"
        self.mock_args.max_new_tokens = 10
        self.mock_args.host = "127.0.0.1"
        self.mock_args.port = 8000

        self.app = create_app(self.mock_model, self.mock_tokenizer, self.mock_args)
        self.client = TestClient(self.app)

    def test_list_models(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "list")
        self.assertEqual(data["data"][0]["id"], "test-model")

    def test_chat_completions_non_streaming(self):
        self.mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        self.mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3]]))

        # Mock model output
        # Output should be input_ids + new tokens
        # Input length is 3. Let's add 2 tokens. Total length 5.
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        self.mock_tokenizer.decode.return_value = "generated text"

        response = self.client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        })

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"]["content"], "generated text")

        # Verify model.generate called
        self.mock_model.generate.assert_called_once()

    def test_completions_non_streaming(self):
        self.mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3]]))
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        self.mock_tokenizer.decode.return_value = "generated text"

        response = self.client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "hello"
        })

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "text_completion")
        self.assertEqual(data["choices"][0]["text"], "generated text")

    @patch("petals.cli.run_http_server.TextIteratorStreamer")
    def test_chat_completions_streaming(self, mock_streamer_cls):
        self.mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        self.mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3]]))

        mock_streamer = MagicMock()
        mock_streamer_cls.return_value = mock_streamer

        # Mock iterator behavior
        mock_streamer.__iter__.return_value = iter(["chunk1", "chunk2"])
        mock_streamer.__next__.side_effect = ["chunk1", "chunk2", StopIteration]

        response = self.client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True
        })

        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")

        # Check SSE format
        self.assertIn("data: {", content)
        self.assertIn("chunk1", content)
        self.assertIn("chunk2", content)
        self.assertIn("data: [DONE]", content)

        # Verify model.generate was called in a separate thread
        time.sleep(0.1)
        self.mock_model.generate.assert_called()

if __name__ == '__main__':
    unittest.main()
