import sys
import unittest.mock
import importlib.util

class MockModule(unittest.mock.MagicMock):
    __path__ = []
    __spec__ = None

sys.modules['hivemind'] = MockModule()
sys.modules['petals.utils'] = MockModule()
sys.modules['petals.utils.misc'] = MockModule()
sys.modules['petals.utils.misc'].DUMMY = unittest.mock.MagicMock()

import torch
import pytest

spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
ptune = importlib.util.module_from_spec(spec)
sys.modules["ptune"] = ptune
spec.loader.exec_module(ptune)

from transformers import PretrainedConfig

class DummyConfig(PretrainedConfig):
    def __init__(self, tuning_mode=None, pre_seq_len=0, hidden_size=64, num_hidden_layers=3):
        super().__init__()
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class MockModel(ptune.PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = unittest.mock.MagicMock()
        self.word_embeddings.weight = unittest.mock.MagicMock()
        self.word_embeddings.weight.device = torch.device("cpu")
        self.word_embeddings.weight.dtype = torch.float32
        self.init_prompts(config)

def test_deep_ptune_shape():
    config = DummyConfig(tuning_mode="deep_ptune", pre_seq_len=10, hidden_size=64, num_hidden_layers=3)
    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(2)
    assert prompts.shape == (2, 10, 64)
    assert intermediate_prompts.shape == (3, 2, 10, 64)
