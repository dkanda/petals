import sys
from unittest.mock import MagicMock
import importlib.util
import pytest
import torch

def pytest_configure():
    class MockPackage(MagicMock):
        __path__ = []
        __spec__ = None

    sys.modules['hivemind'] = MockPackage()
    sys.modules['petals'] = MockPackage()
    sys.modules['petals.utils'] = MockPackage()
    sys.modules['petals.utils.misc'] = MockPackage()
    import petals.utils.misc
    petals.utils.misc.DUMMY = "DUMMY"

pytest_configure()

spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
ptune_module = importlib.util.module_from_spec(spec)
sys.modules["petals.client.ptune"] = ptune_module
spec.loader.exec_module(ptune_module)

from transformers import PretrainedConfig

class MockConfig(PretrainedConfig):
    def __init__(self, tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=3, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class MockModel(torch.nn.Module, ptune_module.PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = torch.nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_intermediate_shape():
    config = MockConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=3)
    model = MockModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # First layer should be zero-padded
    assert torch.all(intermediate_prompts[0] == 0)

def test_ptune_intermediate_shape():
    config = MockConfig(tuning_mode="ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=3)
    model = MockModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert "DUMMY" in repr(intermediate_prompts)
