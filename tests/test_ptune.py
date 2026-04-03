import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import sys
import unittest.mock as mock

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

sys.modules['hivemind'] = MockPackage()
sys.modules['hivemind.utils.logging'] = MockPackage()
sys.modules['hivemind.utils'] = MockPackage()
sys.modules['hivemind.moe.server.module_backend'] = MockPackage()
sys.modules['hivemind.moe.expert_uid'] = MockPackage()
sys.modules['hivemind.utils.nested'] = MockPackage()
sys.modules['hivemind.dht.routing'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.datastructures'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon'] = MockPackage()
sys.modules['tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.slicing_configs'] = MockPackage()

# Load the target module with isolated imports to prevent cascading missing dependencies
import importlib.util
spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
ptune = importlib.util.module_from_spec(spec)
sys.modules["petals.client.ptune"] = ptune
# Mock out internal dependencies to ensure safe loading
sys.modules["petals.utils.misc"] = MockPackage()
sys.modules["petals.utils"] = MockPackage()
sys.modules["petals"] = MockPackage()
spec.loader.exec_module(ptune)

class DummyConfig(PretrainedConfig):
    def __init__(self, tuning_mode="deep_ptune", **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = 5
        self.hidden_size = 16
        self.num_hidden_layers = 4

class DummyModel(nn.Module, ptune.PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_init_prompts():
    config = DummyConfig()
    model = DummyModel(config)

    assert model.pre_seq_len == 5
    assert hasattr(model, 'prompt_embeddings')
    assert hasattr(model, 'intermediate_prompt_embeddings')

    # Check that intermediate embeddings have size (num_hidden_layers - 1)
    assert model.intermediate_prompt_embeddings.weight.shape[1] == (config.num_hidden_layers - 1) * config.hidden_size

def test_ptune_get_prompt():
    config = DummyConfig()
    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # First layer prompt should be zero-padded
    assert torch.all(intermediate_prompts[0] == 0)

if __name__ == "__main__":
    pytest.main([__file__])
