from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from petals.cli.run_http_server import create_app


class MockBatchEncoding(dict):
    def to(self, device):
        return self

    @property
    def input_ids(self):
        return self["input_ids"]


@pytest.fixture
def client():
    # Mock args
    args = MagicMock()
    args.model = "test-model"
    args.token = None
    args.revision = "main"
    args.initial_peers = []
    args.torch_dtype = "auto"
    args.connect_timeout = 10
    args.host = "127.0.0.1"
    args.port = 8000

    with patch("petals.cli.run_http_server.AutoTokenizer") as mock_tokenizer, patch(
        "petals.cli.run_http_server.AutoDistributedModelForCausalLM"
    ) as mock_model:

        # Setup mock tokenizer
        tokenizer_instance = MagicMock()
        tokenizer_instance.apply_chat_template.return_value = "formatted prompt"

        batch_encoding = MockBatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})
        tokenizer_instance.return_value = batch_encoding

        tokenizer_instance.decode.return_value = "generated text"
        mock_tokenizer.from_pretrained.return_value = tokenizer_instance

        # Setup mock model
        model_instance = MagicMock()
        model_instance.device = "cpu"
        model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.from_pretrained.return_value = model_instance

        app = create_app(args)

        with TestClient(app) as c:
            yield c


def test_status(client):
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "test-model"
    assert data["device"] == "cpu"


def test_dashboard(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Petals Server Dashboard" in response.text


def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-model"


def test_chat_completions(client):
    response = client.post(
        "/v1/chat/completions", json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"] == "generated text"


def test_completions(client):
    response = client.post("/v1/completions", json={"model": "test-model", "prompt": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["text"] == "generated text"
