import sys
from unittest.mock import MagicMock
import torch
import pytest
from transformers import PretrainedConfig

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

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
sys.modules['hivemind.dht'] = MockPackage()
sys.modules['hivemind.dht.node'] = MockPackage()
sys.modules['hivemind.dht.routing'] = MockPackage()
sys.modules['hivemind.dht.crypto'] = MockPackage()
sys.modules['hivemind.proto'] = MockPackage()
sys.modules['hivemind.utils'] = MockPackage()
sys.modules['hivemind.utils.logging'] = MockPackage()
sys.modules['hivemind.utils.tensor_deserializer'] = MockPackage()
sys.modules['hivemind.utils.asyncio'] = MockPackage()
sys.modules['hivemind.utils.tensor_descr'] = MockPackage()
sys.modules['hivemind.utils.streaming'] = MockPackage()
sys.modules['hivemind.utils.nested'] = MockPackage()
sys.modules['hivemind.utils.mpfuture'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.control'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.datastructures'] = MockPackage()
sys.modules['hivemind.compression'] = MockPackage()
sys.modules['hivemind.compression.base'] = MockPackage()
sys.modules['hivemind.compression.serialization'] = MockPackage()
sys.modules['tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.slicing_configs'] = MockPackage()
sys.modules['tensor_parallel.tensor_parallel'] = MockPackage()

# petals.utils.misc mocking
sys.modules['petals.utils.misc'] = MagicMock()
sys.modules['petals.utils.misc'].DUMMY = torch.empty(0)

from petals.client.ptune import PTuneMixin, PTuneConfig

class DummyConfig(PretrainedConfig):
    def __init__(self, tuning_mode=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 128
        self.num_hidden_layers = 4
        self.pre_seq_len = 8
        self.tuning_mode = tuning_mode

class DummyModel(PTuneMixin, torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = torch.nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    config = DummyConfig(tuning_mode="ptune")
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 8, 128)
    assert intermediate_prompts is sys.modules['petals.utils.misc'].DUMMY or torch.allclose(intermediate_prompts, sys.modules['petals.utils.misc'].DUMMY)

def test_deep_ptune_shapes():
    config = DummyConfig(tuning_mode="deep_ptune")
    model = DummyModel(config)

    # Assert embedding parameter size
    # intermediate embeddings should be for (num_hidden_layers - 1) layers
    expected_emb_dim = (config.num_hidden_layers - 1) * config.hidden_size
    assert model.intermediate_prompt_embeddings.weight.shape == (8, expected_emb_dim)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 8, 128)
    # The output tensor should be padded to include all layers, even the first one
    assert intermediate_prompts.shape == (config.num_hidden_layers, 2, 8, 128)

    # Check padding of the first layer
    assert torch.all(intermediate_prompts[0] == 0)
