import sys
from unittest.mock import MagicMock

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

# Mock hivemind and its submodules deeply
sys.modules['hivemind'] = MockPackage()
sys.modules['hivemind.moe'] = MockPackage()
sys.modules['hivemind.p2p'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.control'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon'] = MockPackage()
sys.modules['hivemind.proto'] = MockPackage()
sys.modules['hivemind.utils'] = MockPackage()
sys.modules['hivemind.utils.tensor_descr'] = MockPackage()
sys.modules['hivemind.utils.asyncio'] = MockPackage()
sys.modules['hivemind.utils.mpfuture'] = MockPackage()
sys.modules['hivemind.utils.streaming'] = MockPackage()
sys.modules['hivemind.utils.crypto'] = MockPackage()
sys.modules['hivemind.utils.logging'] = MockPackage()
sys.modules['hivemind.utils.nested'] = MockPackage()
sys.modules['hivemind.compression'] = MockPackage()
sys.modules['hivemind.compression.base'] = MockPackage()
sys.modules['hivemind.compression.serialization'] = MockPackage()
sys.modules['hivemind.dht'] = MockPackage()
sys.modules['hivemind.dht.node'] = MockPackage()
sys.modules['hivemind.dht.crypto'] = MockPackage()
sys.modules['hivemind.dht.schema'] = MockPackage()
sys.modules['hivemind.moe.client'] = MockPackage()
sys.modules['hivemind.moe.client.remote_expert_worker'] = MockPackage()
sys.modules['hivemind.moe.expert_uid'] = MockPackage()
sys.modules['hivemind.moe.server'] = MockPackage()
sys.modules['hivemind.moe.server.connection_handler'] = MockPackage()
sys.modules['hivemind.moe.server.module_backend'] = MockPackage()
sys.modules['tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.slicing_configs'] = MockPackage()
sys.modules['tensor_parallel.factory'] = MockPackage()

import torch
import torch.nn as nn
from transformers import PretrainedConfig

# Ensure that the path includes src for petals import
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class DummyConfig(PretrainedConfig):
    def __init__(self, pre_seq_len=0, tuning_mode=None, hidden_size=64, num_hidden_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.pre_seq_len = pre_seq_len
        self.tuning_mode = tuning_mode
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_mode():
    config = DummyConfig(pre_seq_len=10, tuning_mode="ptune", hidden_size=32, num_hidden_layers=3)
    model = DummyModel(config)

    # check param shapes
    assert hasattr(model, 'prompt_embeddings')
    assert not hasattr(model, 'intermediate_prompt_embeddings')
    assert model.prompt_embeddings.weight.shape == (10, 32)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 10, 32)
    assert intermediate_prompts is DUMMY

def test_deep_ptune_mode():
    config = DummyConfig(pre_seq_len=8, tuning_mode="deep_ptune", hidden_size=16, num_hidden_layers=4)
    model = DummyModel(config)

    # check param shapes
    assert hasattr(model, 'prompt_embeddings')
    assert hasattr(model, 'intermediate_prompt_embeddings')
    assert model.prompt_embeddings.weight.shape == (8, 16)

    # intermediate_prompt_embeddings shape should be (pre_seq_len, (num_hidden_layers - 1) * hidden_size)
    assert model.intermediate_prompt_embeddings.weight.shape == (8, 3 * 16)

    prompts, intermediate_prompts = model.get_prompt(batch_size=4)
    assert prompts.shape == (4, 8, 16)

    # intermediate_prompts shape should be (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (4, 4, 8, 16)

    # Verify that the first layer's tensor is zeroed out
    assert torch.all(intermediate_prompts[0] == 0)

    # Verify that the remaining layers' tensors are not zeroed out (unless randomly initialized to zero)
    # Since nn.Embedding is initialized with random normal, it's very unlikely to be exactly zero
    assert not torch.all(intermediate_prompts[1] == 0)
