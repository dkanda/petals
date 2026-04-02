import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import sys
import unittest.mock

class MockPackage(unittest.mock.MagicMock):
    __path__ = []
    __spec__ = None

mock_modules = {
    'hivemind': MockPackage(),
    'hivemind.PeerID': MockPackage(),
    'hivemind.utils': MockPackage(),
    'hivemind.utils.logging': MockPackage(),
    'hivemind.utils.nested': MockPackage(),
    'hivemind.utils.asyncio': MockPackage(),
    'hivemind.utils.streaming': MockPackage(),
    'hivemind.utils.tensor_descr': MockPackage(),
    'hivemind.utils.mpfuture': MockPackage(),
    'hivemind.moe': MockPackage(),
    'hivemind.moe.server': MockPackage(),
    'hivemind.moe.server.connection_handler': MockPackage(),
    'hivemind.moe.server.module_backend': MockPackage(),
    'hivemind.moe.server.expert_uid': MockPackage(),
    'hivemind.moe.client': MockPackage(),
    'hivemind.moe.client.remote_expert_worker': MockPackage(),
    'hivemind.moe.expert_uid': MockPackage(),
    'hivemind.p2p': MockPackage(),
    'hivemind.p2p.p2p_daemon': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings.control': MockPackage(),
    'hivemind.dht': MockPackage(),
    'hivemind.dht.routing': MockPackage(),
    'hivemind.dht.node': MockPackage(),
    'hivemind.proto': MockPackage(),
    'hivemind.proto.runtime_pb2': MockPackage(),
    'hivemind.compression': MockPackage(),
    'hivemind.compression.base': MockPackage(),
    'hivemind.compression.serialization': MockPackage(),
    'hivemind.optim': MockPackage(),
    'hivemind.optim.performance_ema': MockPackage(),
    'tensor_parallel': MockPackage(),
    'tensor_parallel.slicing_configs': MockPackage(),
    'tensor_parallel.tensor_parallel': MockPackage(),
}

def setup_module():
    global patcher
    patcher = unittest.mock.patch.dict('sys.modules', mock_modules)
    patcher.start()

def teardown_module():
    patcher.stop()

class MockWordEmbeddings(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(10, hidden_size))

def get_mock_model(config):
    from petals.client.ptune import PTuneMixin
    class MockModel(PTuneMixin, nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = MockWordEmbeddings(config.hidden_size)
            self.init_prompts(config)
    return MockModel(config)

def test_ptune_mode():
    from petals.utils.misc import DUMMY

    config = PretrainedConfig()
    config.tuning_mode = "ptune"
    config.pre_seq_len = 5
    config.hidden_size = 16
    config.num_hidden_layers = 4

    model = get_mock_model(config)
    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts is DUMMY or torch.allclose(intermediate_prompts, DUMMY)

def test_deep_ptune_mode():
    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5
    config.hidden_size = 16
    config.num_hidden_layers = 4

    model = get_mock_model(config)
    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")

    # Check that intermediate prompt embeddings parameter allocates only num_hidden_layers - 1
    assert model.intermediate_prompt_embeddings.weight.shape == (5, 3 * 16)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)

    # Check that output is prepended with zeros and shape is num_hidden_layers x batch_size x pre_seq_len x hidden_size
    assert intermediate_prompts.shape == (4, 2, 5, 16)

    # Verify first layer is zeroes
    assert torch.all(intermediate_prompts[0] == 0)

    # Verify other layers are not zeroes (unless by chance, but initialized from random)
    assert not torch.all(intermediate_prompts[1:] == 0)
