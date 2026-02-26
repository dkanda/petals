import sys
import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import importlib

@pytest.fixture(scope="module")
def server_module():
    # Create extensive mocks for hivemind
    mock_hivemind = MagicMock()

    mocks = {
        "hivemind": mock_hivemind,
        "hivemind.moe": MagicMock(),
        "hivemind.moe.expert_uid": MagicMock(),
        "hivemind.moe.client": MagicMock(),
        "hivemind.moe.client.remote_expert_worker": MagicMock(),
        "hivemind.moe.server": MagicMock(),
        "hivemind.moe.server.connection_handler": MagicMock(),
        "hivemind.moe.server.module_backend": MagicMock(),
        "hivemind.utils": MagicMock(),
        "hivemind.utils.logging": MagicMock(),
        "hivemind.utils.mpfuture": MagicMock(),
        "hivemind.utils.asyncio": MagicMock(),
        "hivemind.utils.streaming": MagicMock(),
        "hivemind.utils.crypto": MagicMock(),
        "hivemind.utils.nested": MagicMock(),
        "hivemind.utils.tensor_descr": MagicMock(),
        "hivemind.utils.serializer": MagicMock(),
        "hivemind.utils.networking": MagicMock(),
        "hivemind.utils.timed_storage": MagicMock(),
        "hivemind.proto": MagicMock(),
        "hivemind.proto.runtime_pb2": MagicMock(),
        "hivemind.dht": MagicMock(),
        "hivemind.dht.node": MagicMock(),
        "hivemind.server": MagicMock(),
        "hivemind.server.expert_backend": MagicMock(),
        "hivemind.optim": MagicMock(),
        "hivemind.averaging": MagicMock(),
        "hivemind.p2p": MagicMock(),
        "hivemind.p2p.p2p_daemon": MagicMock(),
        "hivemind.p2p.p2p_daemon_bindings": MagicMock(),
        "hivemind.p2p.p2p_daemon_bindings.control": MagicMock(),
        "hivemind.compression": MagicMock(),
        "hivemind.compression.serialization": MagicMock(),
    }

    # Patch sys.modules with mocks
    with patch.dict(sys.modules, mocks):
        # Add src to path
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
        if src_path not in sys.path:
            sys.path.append(src_path)

        # Import the module
        # We need to make sure we don't use a cached version that might have failed or used real modules
        import petals.cli.run_http_server as module
        importlib.reload(module)

        yield module

@pytest.fixture(autouse=True)
def reset_globals(server_module):
    server_module.model = None
    server_module.tokenizer = None
    server_module.model_name = None
    yield
    server_module.model = None
    server_module.tokenizer = None
    server_module.model_name = None

@pytest.fixture
def client(server_module):
    return TestClient(server_module.app)

@pytest.fixture
def mock_engine(server_module):
    server_module.model_name = "test-model"

    mock_tokenizer = MagicMock()
    # Mock apply_chat_template to return a string
    mock_tokenizer.apply_chat_template.return_value = "User: Hello\nAssistant:"
    # Mock tokenizer call to return a dict with input_ids
    mock_input = MagicMock()
    mock_input.shape = (1, 5) # Batch size 1, sequence length 5
    # Configure len(inputs[0])
    mock_row = MagicMock()
    mock_row.__len__.return_value = 5
    mock_input.__getitem__.return_value = mock_row

    mock_tokenizer.return_value = {"input_ids": mock_input}

    # Mock decode
    mock_tokenizer.decode.return_value = "User: Hello\nAssistant: Hi there"

    mock_model = MagicMock()
    # Mock generate return
    # Generate returns [sequences]
    # For batch size 1, sequence length 5 (input) + 2 (output) = 7
    mock_output = MagicMock()
    mock_output.__len__.return_value = 7
    mock_model.generate.return_value = [mock_output]

    server_module.tokenizer = mock_tokenizer
    server_module.model = mock_model

    return mock_tokenizer, mock_model

def test_list_models(client, server_module):
    server_module.model_name = "test-model"
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "test-model"

def test_chat_completions(client, mock_engine, server_module):
    mock_tokenizer, mock_model = mock_engine

    mock_tokenizer.decode.return_value = "Hi there"

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}]
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hi there"
    # Check usage
    assert data["usage"]["prompt_tokens"] == 5
    assert data["usage"]["completion_tokens"] == 2 # 7 - 5
    assert data["usage"]["total_tokens"] == 7

def test_completions(client, mock_engine, server_module):
    mock_tokenizer, mock_model = mock_engine

    mock_tokenizer.decode.return_value = "User: Hello\nAssistant: Hi there"

    prompt = "User: Hello\nAssistant:"

    response = client.post("/v1/completions", json={
        "model": "test-model",
        "prompt": prompt
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert data["choices"][0]["text"] == " Hi there"

def test_invalid_model(client, server_module):
    server_module.model_name = "test-model"
    response = client.post("/v1/chat/completions", json={
        "model": "wrong-model",
        "messages": []
    })
    assert response.status_code == 404
