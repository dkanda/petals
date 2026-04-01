import sys
from unittest.mock import MagicMock

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

# Mock external dependencies that are difficult to install or unnecessary for these tests
sys.modules['hivemind'] = MockPackage()
sys.modules['hivemind.moe'] = MockPackage()
sys.modules['hivemind.moe.expert_uid'] = MockPackage()
sys.modules['hivemind.moe.client'] = MockPackage()
sys.modules['hivemind.moe.client.remote_expert_worker'] = MockPackage()
sys.modules['hivemind.moe.server'] = MockPackage()
sys.modules['hivemind.moe.server.module_backend'] = MockPackage()
sys.modules['hivemind.moe.server.connection_handler'] = MockPackage()
sys.modules['hivemind.p2p'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.control'] = MockPackage()
sys.modules['hivemind.utils'] = MockPackage()
sys.modules['hivemind.utils.nested'] = MockPackage()
sys.modules['hivemind.utils.logging'] = MockPackage()
sys.modules['hivemind.utils.asyncio'] = MockPackage()
sys.modules['hivemind.utils.tensor_descr'] = MockPackage()
sys.modules['hivemind.utils.mpfuture'] = MockPackage()
sys.modules['hivemind.utils.streaming'] = MockPackage()
sys.modules['hivemind.dht'] = MockPackage()
sys.modules['hivemind.dht.node'] = MockPackage()
sys.modules['hivemind.dht.routing'] = MockPackage()
sys.modules['hivemind.compression'] = MockPackage()
sys.modules['hivemind.compression.serialization'] = MockPackage()
sys.modules['hivemind.proto'] = MockPackage()
sys.modules['tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.slicing_configs'] = MockPackage()

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class DummyWordEmbeddings:
    def __init__(self, hidden_size):
        self.weight = torch.zeros((100, hidden_size))

class DummyConfig(PretrainedConfig):
    def __init__(self, tuning_mode=None, pre_seq_len=0, num_hidden_layers=4, hidden_size=8, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = DummyWordEmbeddings(config.hidden_size)
        self.init_prompts(config)

def test_ptune_mode():
    config = DummyConfig(tuning_mode="ptune", pre_seq_len=5, num_hidden_layers=4, hidden_size=8)
    model = DummyModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts is DUMMY or torch.allclose(intermediate_prompts, DUMMY)

def test_deep_ptune_mode():
    config = DummyConfig(tuning_mode="deep_ptune", pre_seq_len=5, num_hidden_layers=4, hidden_size=8)
    model = DummyModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")

    # Verify the parameter size is reduced by 1 layer
    assert model.intermediate_prompt_embeddings.weight.shape == (
        config.pre_seq_len,
        (config.num_hidden_layers - 1) * config.hidden_size
    )

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # Verify intermediate prompts shape
    assert intermediate_prompts.shape == (
        config.num_hidden_layers,
        batch_size,
        config.pre_seq_len,
        config.hidden_size
    )

    # Verify the first layer is zero-padded
    zero_layer = intermediate_prompts[0]
    assert torch.all(zero_layer == 0)

    # Ensure remaining layers come from embeddings and might not be zero (after init)
    # The chance of them being exactly zero by random init is practically zero
    remaining_layers = intermediate_prompts[1:]
    assert remaining_layers.shape == (
        config.num_hidden_layers - 1,
        batch_size,
        config.pre_seq_len,
        config.hidden_size
    )

def test_no_tuning_mode():
    config = DummyConfig(tuning_mode=None, pre_seq_len=0, num_hidden_layers=4, hidden_size=8)
    model = DummyModel(config)

    assert not hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")
