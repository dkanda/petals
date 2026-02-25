import json
import threading
import unittest
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from petals.cli import run_http_server
from petals.cli.run_http_server import app


class TestHTTPServer(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()

        # Mock tokenizer encode/decode
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        self.mock_tokenizer.decode.return_value = "generated text"
        self.mock_tokenizer.chat_template = None

        # Mock model generate
        self.mock_model.device = "cpu"
        self.mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

        # Set up args
        run_http_server.args = MagicMock()
        run_http_server.args.model = "test-model"
        run_http_server.args.initial_peers = []
        run_http_server.args.torch_dtype = "float32"
        run_http_server.args.token = None

    @patch("petals.cli.run_http_server.AutoDistributedModelForCausalLM")
    @patch("petals.cli.run_http_server.AutoTokenizer")
    def test_models_endpoint(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = self.mock_model
        mock_tokenizer_cls.from_pretrained.return_value = self.mock_tokenizer

        with TestClient(app) as client:
            response = client.get("/v1/models")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["object"], "list")
            self.assertEqual(data["data"][0]["id"], "test-model")

    @patch("petals.cli.run_http_server.AutoDistributedModelForCausalLM")
    @patch("petals.cli.run_http_server.AutoTokenizer")
    def test_chat_completions(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = self.mock_model
        mock_tokenizer_cls.from_pretrained.return_value = self.mock_tokenizer

        # Mock TextIteratorStreamer to return some tokens
        mock_streamer = MagicMock()
        mock_streamer.__iter__.return_value = iter(["generated", " ", "text"])

        with patch("petals.cli.run_http_server.TextIteratorStreamer", return_value=mock_streamer):
            with TestClient(app) as client:
                payload = {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False
                }
                response = client.post("/v1/chat/completions", json=payload)
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["choices"][0]["message"]["content"], "generated text")

    @patch("petals.cli.run_http_server.AutoDistributedModelForCausalLM")
    @patch("petals.cli.run_http_server.AutoTokenizer")
    def test_completions(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = self.mock_model
        mock_tokenizer_cls.from_pretrained.return_value = self.mock_tokenizer

        # Mock TextIteratorStreamer
        mock_streamer = MagicMock()
        mock_streamer.__iter__.return_value = iter(["generated", " ", "text"])

        with patch("petals.cli.run_http_server.TextIteratorStreamer", return_value=mock_streamer):
            with TestClient(app) as client:
                payload = {
                    "model": "test-model",
                    "prompt": "Once upon a time",
                    "stream": False
                }
                response = client.post("/v1/completions", json=payload)
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["choices"][0]["text"], "generated text")

    @patch("petals.cli.run_http_server.AutoDistributedModelForCausalLM")
    @patch("petals.cli.run_http_server.AutoTokenizer")
    def test_chat_completions_streaming(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = self.mock_model
        mock_tokenizer_cls.from_pretrained.return_value = self.mock_tokenizer

        mock_streamer = MagicMock()
        mock_streamer.__iter__.return_value = iter(["generated", " ", "text"])

        with patch("petals.cli.run_http_server.TextIteratorStreamer", return_value=mock_streamer):
            with TestClient(app) as client:
                payload = {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True
                }
                response = client.post("/v1/chat/completions", json=payload)
                self.assertEqual(response.status_code, 200)
                # Verify SSE format
                content = response.text
                self.assertIn("data: {", content)
                self.assertIn("[DONE]", content)
