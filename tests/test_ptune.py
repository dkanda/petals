import sys
import unittest.mock
from unittest.mock import MagicMock

class MockPackage:
    def __init__(self, name=""):
        self.__name__ = name
        self.__path__ = []
        self.__spec__ = None

    def __getattr__(self, name):
        if name in ("__path__", "__spec__"):
            raise AttributeError()
        mock = MagicMock()
        mock.__name__ = name
        return mock

def setup_module():
    mock_modules = {
        "hivemind": MockPackage("hivemind"),
        "hivemind.p2p": MockPackage("hivemind.p2p"),
        "hivemind.p2p.p2p_daemon_bindings": MockPackage("hivemind.p2p.p2p_daemon_bindings"),
        "hivemind.p2p.p2p_daemon_bindings.control": MockPackage("hivemind.p2p.p2p_daemon_bindings.control"),
        "hivemind.p2p.p2p_daemon": MockPackage("hivemind.p2p.p2p_daemon"),
        "hivemind.utils": MockPackage("hivemind.utils"),
        "hivemind.utils.nested": MockPackage("hivemind.utils.nested"),
        "hivemind.utils.tensor_descr": MockPackage("hivemind.utils.tensor_descr"),
        "hivemind.utils.asyncio": MockPackage("hivemind.utils.asyncio"),
        "hivemind.utils.logging": MockPackage("hivemind.utils.logging"),
        "hivemind.utils.mpfuture": MockPackage("hivemind.utils.mpfuture"),
        "hivemind.utils.streaming": MockPackage("hivemind.utils.streaming"),
        "hivemind.dht": MockPackage("hivemind.dht"),
        "hivemind.dht.routing": MockPackage("hivemind.dht.routing"),
        "hivemind.dht.node": MockPackage("hivemind.dht.node"),
        "hivemind.dht.crypto": MockPackage("hivemind.dht.crypto"),
        "hivemind.dht.schema": MockPackage("hivemind.dht.schema"),
        "hivemind.dht.validation": MockPackage("hivemind.dht.validation"),
        "hivemind.moe": MockPackage("hivemind.moe"),
        "hivemind.moe.server": MockPackage("hivemind.moe.server"),
        "hivemind.moe.server.connection_handler": MockPackage("hivemind.moe.server.connection_handler"),
        "hivemind.moe.server.module_backend": MockPackage("hivemind.moe.server.module_backend"),
        "hivemind.moe.server.task_pool": MockPackage("hivemind.moe.server.task_pool"),
        "hivemind.moe.expert_uid": MockPackage("hivemind.moe.expert_uid"),
        "hivemind.moe.client": MockPackage("hivemind.moe.client"),
        "hivemind.moe.client.remote_expert_worker": MockPackage("hivemind.moe.client.remote_expert_worker"),
        "hivemind.moe.client.expert": MockPackage("hivemind.moe.client.expert"),
        "hivemind.proto": MockPackage("hivemind.proto"),
        "hivemind.proto.runtime_pb2": MockPackage("hivemind.proto.runtime_pb2"),
        "hivemind.compression": MockPackage("hivemind.compression"),
        "hivemind.compression.serialization": MockPackage("hivemind.compression.serialization"),
        "hivemind.compression.base": MockPackage("hivemind.compression.base"),
        "tensor_parallel": MockPackage("tensor_parallel"),
        "tensor_parallel.slicing_configs": MockPackage("tensor_parallel.slicing_configs"),
        "tensor_parallel.tensor_parallel": MockPackage("tensor_parallel.tensor_parallel"),
        "tensor_parallel.cross_device_ops": MockPackage("tensor_parallel.cross_device_ops"),
    }
    # Using patch.dict to mock globally for this module context
    global _patcher
    _patcher = unittest.mock.patch.dict("sys.modules", mock_modules)
    _patcher.start()

def teardown_module():
    _patcher.stop()

# Ensure the mock is setup before we import petals modules
setup_module()
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin, force_non_empty_weights
from petals.utils.misc import DUMMY

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)

def test_ptune_mode():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="ptune",
        pre_seq_len=5
    )
    model = MockModel(config)
    model.init_prompts(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, 5, 16)
    assert intermediate_prompts is DUMMY

def test_deep_ptune_mode():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=5
    )
    model = MockModel(config)
    model.init_prompts(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")

    # Check that intermediate prompt embeddings shape only covers num_hidden_layers - 1
    assert model.intermediate_prompt_embeddings.weight.shape == (5, (4 - 1) * 16)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Output shapes expected:
    # prompts: (batch_size, pre_seq_len, hidden_size)
    # intermediate_prompts: (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert prompts.shape == (batch_size, 5, 16)
    assert intermediate_prompts.shape == (4, batch_size, 5, 16)

    # Assert the prepended padding for the first layer is all zeros
    assert torch.all(intermediate_prompts[0] == 0)
