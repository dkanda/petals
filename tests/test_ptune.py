import pytest
import sys
from unittest.mock import MagicMock

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

sys.modules["hivemind"] = MockPackage()
sys.modules["hivemind.p2p.p2p_daemon_bindings"] = MockPackage()
sys.modules["hivemind.moe"] = MockPackage()
sys.modules["hivemind.moe.client"] = MockPackage()
sys.modules["hivemind.moe.client.remote_expert_worker"] = MockPackage()
sys.modules["hivemind.compression"] = MockPackage()
sys.modules["hivemind.compression.serialization"] = MockPackage()
sys.modules["hivemind.utils"] = MockPackage()
sys.modules["hivemind.utils.nested"] = MockPackage()
sys.modules["hivemind.dht"] = MockPackage()
sys.modules["hivemind.dht.routing"] = MockPackage()
sys.modules["hivemind.dht.node"] = MockPackage()
sys.modules["hivemind.p2p"] = MockPackage()
sys.modules["hivemind.p2p.p2p_daemon"] = MockPackage()
sys.modules["hivemind.p2p.p2p_daemon_bindings.control"] = MockPackage()
sys.modules["hivemind.utils.asyncio"] = MockPackage()
sys.modules["hivemind.proto"] = MockPackage()
sys.modules["hivemind.utils.tensor_descr"] = MockPackage()
sys.modules["hivemind.utils.streaming"] = MockPackage()
sys.modules["hivemind.utils.logging"] = MockPackage()
sys.modules["hivemind.moe.server"] = MockPackage()
sys.modules["hivemind.moe.server.connection_handler"] = MockPackage()
sys.modules["hivemind.moe.server.module_backend"] = MockPackage()
sys.modules["hivemind.moe.expert_uid"] = MockPackage()
sys.modules["tensor_parallel"] = MockPackage()
sys.modules["hivemind.utils.mpfuture"] = MockPackage()
sys.modules["tensor_parallel.slicing_configs"] = MockPackage()
sys.modules["tensor_parallel.tensor_parallel"] = MockPackage()

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin

class DummyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 2
        self.hidden_size = 16
        self.num_hidden_layers = 4

class DummyPTuneModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_get_prompt_shapes_and_zeros():
    config = DummyConfig()
    model = DummyPTuneModel(config)

    batch_size = 1
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Assert shapes
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # Verify the first layer of intermediate prompts is zero padded
    first_layer_prompts = intermediate_prompts[0]
    expected_zeros = torch.zeros(
        batch_size,
        config.pre_seq_len,
        config.hidden_size,
        dtype=intermediate_prompts.dtype,
        device=intermediate_prompts.device
    )
    assert torch.allclose(first_layer_prompts, expected_zeros), "First layer of deep_ptune intermediate prompts must be zero-padded."

def test_ptune_only_get_prompt():
    config = DummyConfig()
    config.tuning_mode = "ptune"
    model = DummyPTuneModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    # The intermediate_prompts should be DUMMY in standard "ptune" mode.
    # DUMMY is from petals.utils.misc
    from petals.utils.misc import DUMMY
    assert intermediate_prompts is DUMMY
