import sys
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from transformers import PretrainedConfig

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

hivemind_mock = MockPackage()
sys.modules["hivemind"] = hivemind_mock
sys.modules["hivemind.moe"] = MockPackage()
sys.modules["hivemind.moe.client"] = MockPackage()
sys.modules["hivemind.moe.client.remote_expert_worker"] = MockPackage()
sys.modules["hivemind.moe.expert_uid"] = MockPackage()
sys.modules["hivemind.moe.server"] = MockPackage()
sys.modules["hivemind.moe.server.connection_handler"] = MockPackage()
sys.modules["hivemind.moe.server.module_backend"] = MockPackage()
sys.modules["hivemind.p2p"] = MockPackage()
sys.modules["hivemind.p2p.p2p_daemon"] = MockPackage()
sys.modules["hivemind.p2p.p2p_daemon_bindings"] = MockPackage()
sys.modules["hivemind.p2p.p2p_daemon_bindings.control"] = MockPackage()
sys.modules["hivemind.utils"] = MockPackage()
sys.modules["hivemind.utils.logging"] = MockPackage()
sys.modules["hivemind.utils.mpfuture"] = MockPackage()
sys.modules["hivemind.compression"] = MockPackage()
sys.modules["hivemind.compression.base"] = MockPackage()
sys.modules["hivemind.compression.serialization"] = MockPackage()
sys.modules["hivemind.proto"] = MockPackage()
sys.modules["hivemind.dht"] = MockPackage()
sys.modules["hivemind.dht.node"] = MockPackage()
sys.modules["hivemind.dht.crypto"] = MockPackage()
sys.modules["hivemind.dht.validation"] = MockPackage()
sys.modules["hivemind.utils.networking"] = MockPackage()
sys.modules["hivemind.utils.asyncio"] = MockPackage()
sys.modules["hivemind.utils.tensor_deserialization"] = MockPackage()
sys.modules["hivemind.utils.streaming"] = MockPackage()
sys.modules["hivemind.utils.nested"] = MockPackage()
sys.modules["hivemind.utils.tensor_descr"] = MockPackage()
sys.modules["hivemind.utils.timed_storage"] = MockPackage()
sys.modules["hivemind.utils.performance_ema"] = MockPackage()
sys.modules["tensor_parallel"] = MockPackage()
sys.modules["tensor_parallel.tensor_parallel"] = MockPackage()
sys.modules["tensor_parallel.slicing_configs"] = MockPackage()

import petals.client.ptune as ptune
from petals.client.ptune import PTuneMixin

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=4,
        tuning_mode="ptune",
        pre_seq_len=10,
    )
    model = MockModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    # DUMMY is usually empty tensor when casted to dtype
    assert isinstance(intermediate_prompts, torch.Tensor)
    assert intermediate_prompts.numel() == 0

def test_deep_ptune_shapes():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=10,
    )
    model = MockModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (
        config.num_hidden_layers,
        batch_size,
        config.pre_seq_len,
        config.hidden_size,
    )

    # Check that the first layer prompt is zero-padded
    assert torch.all(intermediate_prompts[0] == 0)
