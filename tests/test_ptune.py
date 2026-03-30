import sys
from unittest.mock import MagicMock

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

# Mock hivemind and other difficult imports
sys.modules['hivemind'] = MockPackage()
sys.modules['hivemind.moe'] = MockPackage()
sys.modules['hivemind.moe.client'] = MockPackage()
sys.modules['hivemind.moe.client.remote_expert_worker'] = MockPackage()
sys.modules['hivemind.moe.expert_uid'] = MockPackage()
sys.modules['hivemind.moe.server'] = MockPackage()
sys.modules['hivemind.moe.server.connection_handler'] = MockPackage()
sys.modules['hivemind.moe.server.module_backend'] = MockPackage()
sys.modules['hivemind.p2p'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.control'] = MockPackage()
sys.modules['hivemind.proto'] = MockPackage()
sys.modules['hivemind.utils'] = MockPackage()
sys.modules['hivemind.utils.asyncio'] = MockPackage()
sys.modules['hivemind.utils.logging'] = MockPackage()
sys.modules['hivemind.utils.mpfuture'] = MockPackage()
sys.modules['hivemind.utils.nested'] = MockPackage()
sys.modules['hivemind.utils.streaming'] = MockPackage()
sys.modules['hivemind.utils.tensor_descr'] = MockPackage()
sys.modules['hivemind.compression'] = MockPackage()
sys.modules['hivemind.compression.base'] = MockPackage()
sys.modules['hivemind.compression.serialization'] = MockPackage()
sys.modules['hivemind.dht'] = MockPackage()
sys.modules['hivemind.dht.node'] = MockPackage()
sys.modules['hivemind.dht.routing'] = MockPackage()
sys.modules['hivemind.dht.crypto'] = MockPackage()
sys.modules['tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.slicing_configs'] = MockPackage()
sys.modules['tensor_parallel.tensor_parallel'] = MockPackage()

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin, PTuneConfig

class MockConfig(PretrainedConfig):
    def __init__(self, pre_seq_len=0, tuning_mode=None, hidden_size=128, num_hidden_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.pre_seq_len = pre_seq_len
        self.tuning_mode = tuning_mode
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class MockModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_initialization():
    config = MockConfig(pre_seq_len=10, tuning_mode="ptune")
    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")
    assert model.prompt_embeddings.weight.shape == (10, 128)

def test_deep_ptune_initialization():
    config = MockConfig(pre_seq_len=10, tuning_mode="deep_ptune", num_hidden_layers=4)
    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")

    # Check that it creates embeddings for num_hidden_layers - 1
    assert model.intermediate_prompt_embeddings.weight.shape == (10, (4 - 1) * 128)

def test_get_prompt_ptune():
    config = MockConfig(pre_seq_len=10, tuning_mode="ptune")
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (2, 10, 128)
    # PTune doesn't have intermediate prompts, returns DUMMY string from misc.py
    # But checking type is enough or check if it equals DUMMY
    from petals.utils.misc import DUMMY
    assert intermediate_prompts is DUMMY or torch.allclose(intermediate_prompts, DUMMY)

def test_get_prompt_deep_ptune():
    config = MockConfig(pre_seq_len=10, tuning_mode="deep_ptune", num_hidden_layers=4, hidden_size=128)
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, 10, 128)

    # Shape should be (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (4, batch_size, 10, 128)

    # Check that first layer is all zeros
    first_layer = intermediate_prompts[0]
    assert torch.all(first_layer == 0)
