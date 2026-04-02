import sys
import unittest.mock
import pytest
import torch
import torch.nn as nn

class MockPackage:
    __path__ = []
    __spec__ = None
    __name__ = "mock"
    def __getattr__(self, item):
        return unittest.mock.Mock()

mock_modules = {
    'hivemind': MockPackage(),
    'hivemind.p2p': MockPackage(),
    'hivemind.p2p.p2p_daemon': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings.control': MockPackage(),
    'hivemind.utils': MockPackage(),
    'hivemind.utils.nested': MockPackage(),
    'hivemind.utils.tensor_descr': MockPackage(),
    'hivemind.utils.logging': MockPackage(),
    'hivemind.utils.asyncio': MockPackage(),
    'hivemind.utils.streaming': MockPackage(),
    'hivemind.utils.mpfuture': MockPackage(),
    'hivemind.moe': MockPackage(),
    'hivemind.moe.expert_uid': MockPackage(),
    'hivemind.moe.client': MockPackage(),
    'hivemind.moe.client.remote_expert_worker': MockPackage(),
    'hivemind.moe.server': MockPackage(),
    'hivemind.moe.server.connection_handler': MockPackage(),
    'hivemind.moe.server.module_backend': MockPackage(),
    'hivemind.dht': MockPackage(),
    'hivemind.dht.routing': MockPackage(),
    'hivemind.dht.node': MockPackage(),
    'hivemind.compression': MockPackage(),
    'hivemind.compression.serialization': MockPackage(),
    'hivemind.proto': MockPackage(),
    'hivemind.proto.runtime_pb2': MockPackage(),
    'tensor_parallel': MockPackage(),
    'tensor_parallel.tensor_parallel': MockPackage(),
    'tensor_parallel.slicing_configs': MockPackage(),
}

import sys
sys.modules.update(mock_modules)

from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune():
    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )
    model = MockModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts is DUMMY or torch.allclose(intermediate_prompts, DUMMY)
    assert prompts.dtype == torch.float32

def test_deep_ptune():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )
    model = MockModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)
    assert prompts.dtype == torch.float32
    assert intermediate_prompts.dtype == torch.float32

    # Check that first layer padding is all zeros
    assert torch.allclose(intermediate_prompts[0], torch.zeros_like(intermediate_prompts[0]))

    # Ensure other layers are not all zeros (randomly initialized via Embedding)
    assert not torch.allclose(intermediate_prompts[1:], torch.zeros_like(intermediate_prompts[1:]))
