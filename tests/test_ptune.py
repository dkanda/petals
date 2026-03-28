import sys
from unittest.mock import MagicMock

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

sys.modules["hivemind"] = MockPackage()
sys.modules["hivemind.moe"] = MockPackage()
sys.modules["hivemind.p2p"] = MockPackage()
sys.modules["hivemind.proto"] = MockPackage()
sys.modules["hivemind.utils"] = MockPackage()
sys.modules["hivemind.compression"] = MockPackage()
sys.modules["hivemind.dht"] = MockPackage()
sys.modules["hivemind.moe.client"] = MockPackage()
sys.modules["hivemind.moe.client.remote_expert_worker"] = MockPackage()
sys.modules["hivemind.utils.tensor_descr"] = MockPackage()
sys.modules["hivemind.utils.logging"] = MockPackage()
sys.modules["hivemind.moe.expert_uid"] = MockPackage()
sys.modules["hivemind.moe.server"] = MockPackage()
sys.modules["hivemind.moe.server.connection_handler"] = MockPackage()
sys.modules["hivemind.moe.server.module_backend"] = MockPackage()
sys.modules["hivemind.p2p.p2p_daemon"] = MockPackage()
sys.modules["hivemind.p2p.p2p_daemon_bindings"] = MockPackage()
sys.modules["hivemind.p2p.p2p_daemon_bindings.control"] = MockPackage()
sys.modules["hivemind.utils.asyncio"] = MockPackage()
sys.modules["hivemind.utils.streaming"] = MockPackage()
sys.modules["hivemind.utils.mpfuture"] = MockPackage()
sys.modules["hivemind.utils.nested"] = MockPackage()
sys.modules["hivemind.compression.serialization"] = MockPackage()
sys.modules["hivemind.dht.node"] = MockPackage()
sys.modules["tensor_parallel"] = MockPackage()
sys.modules["tensor_parallel.slicing_configs"] = MockPackage()
sys.modules["tensor_parallel.tensor_parallel"] = MockPackage()

import torch
import torch.nn as nn
from transformers import PretrainedConfig
import pytest

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class MockWordEmbeddings:
    def __init__(self, hidden_size):
        self.weight = torch.randn(10, hidden_size)

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = MockWordEmbeddings(config.hidden_size)
        self.init_prompts(config)

def test_ptune_initialization():
    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4
    )
    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")

    # Check shape of prompt_embeddings
    assert model.prompt_embeddings.weight.shape == (5, 16)

def test_ptune_get_prompt():
    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4
    )
    model = MockModel(config)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts is DUMMY

def test_deep_ptune_initialization():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4
    )
    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")

    # Check shapes
    assert model.prompt_embeddings.weight.shape == (5, 16)
    # Intermediate should be num_hidden_layers - 1
    assert model.intermediate_prompt_embeddings.weight.shape == (5, 3 * 16)

def test_deep_ptune_get_prompt():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4
    )
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, 5, 16)

    # Total shape should be (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (4, batch_size, 5, 16)

    # The first layer should be zero-padded
    assert torch.all(intermediate_prompts[0] == 0)

    # The other layers should contain some data from the embeddings
    assert not torch.all(intermediate_prompts[1:] == 0)
