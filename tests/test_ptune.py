import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from transformers import PretrainedConfig
import sys

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

mock_modules = {
    'hivemind': MockPackage(),
    'hivemind.p2p': MockPackage(),
    'hivemind.p2p.p2p_daemon': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings.datastructures': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings.control': MockPackage(),
    'hivemind.dht': MockPackage(),
    'hivemind.dht.node': MockPackage(),
    'hivemind.moe': MockPackage(),
    'hivemind.moe.client': MockPackage(),
    'hivemind.moe.expert_uid': MockPackage(),
    'hivemind.moe.client.remote_expert_worker': MockPackage(),
    'hivemind.moe.server': MockPackage(),
    'hivemind.moe.server.connection_handler': MockPackage(),
    'hivemind.moe.server.module_backend': MockPackage(),
    'hivemind.utils': MockPackage(),
    'hivemind.utils.nested': MockPackage(),
    'hivemind.utils.tensor_descr': MockPackage(),
    'hivemind.utils.logging': MockPackage(),
    'hivemind.utils.asyncio': MockPackage(),
    'hivemind.utils.streaming': MockPackage(),
    'hivemind.utils.mpfuture': MockPackage(),
    'hivemind.dht.routing': MockPackage(),
    'hivemind.compression': MockPackage(),
    'hivemind.compression.serialization': MockPackage(),
    'hivemind.proto': MockPackage(),
    'hivemind.proto.runtime_pb2': MockPackage(),
    'hivemind.proto.dht_pb2': MockPackage(),
    'tensor_parallel': MockPackage(),
    'tensor_parallel.slicing_configs': MockPackage(),
    'tensor_parallel.tensor_parallel': MockPackage(),
}

def setup_module():
    sys.modules.update(mock_modules)

def teardown_module():
    for mod in mock_modules:
        sys.modules.pop(mod, None)

class DummyModel(nn.Module):
    def __init__(self, config):
        from petals.client.ptune import PTuneMixin
        self.__class__.__bases__ = (nn.Module, PTuneMixin)
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    from petals.utils.misc import DUMMY
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=4,
    )
    config.tuning_mode = "ptune"
    config.pre_seq_len = 10

    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts is DUMMY or torch.allclose(intermediate_prompts, DUMMY)

def test_deep_ptune_shapes():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=4,
    )
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 10

    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # First layer prompt should be zero-padded
    assert torch.all(intermediate_prompts[0] == 0)
