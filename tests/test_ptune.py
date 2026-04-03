import sys
import unittest.mock as mock

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

sys.modules['tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.slicing_configs'] = MockPackage()
sys.modules['hivemind'] = MockPackage()

import torch
import pytest
from transformers import PretrainedConfig
import importlib.util

spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
ptune_module = importlib.util.module_from_spec(spec)

class Dummy:
    pass
DUMMY = Dummy()
sys.modules["petals"] = MockPackage()
sys.modules["petals.utils"] = MockPackage()
sys.modules["petals.utils.misc"] = MockPackage()
sys.modules["petals.utils.misc"].DUMMY = DUMMY

spec.loader.exec_module(ptune_module)
PTuneMixin = ptune_module.PTuneMixin

class MockWordEmbeddings:
    weight = torch.zeros(1, dtype=torch.float32, device="cpu")

class MockConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 10
        self.num_hidden_layers = 4
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 5

class MockModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = MockWordEmbeddings()
        self.init_prompts(config)

def test_deep_ptune():
    config = MockConfig()
    model = MockModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    # The intermediate prompts should have shape: (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
