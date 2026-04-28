import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import sys
import unittest.mock as mock

class DummyConfig(PretrainedConfig):
    def __init__(self, tuning_mode=None, pre_seq_len=0, num_hidden_layers=4, hidden_size=16, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

# Mock hivemind as described in memory
class MockHivemind(mock.MagicMock):
    __path__ = []
    __spec__ = None

sys.modules['hivemind'] = MockHivemind()

def test_deep_ptune():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)
    sys.modules["ptune"] = ptune
    with mock.patch.dict('sys.modules', {'hivemind': MockHivemind()}):
        spec.loader.exec_module(ptune)

    PTuneMixin = ptune.PTuneMixin
    class DummyModel(nn.Module, PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

    config = DummyConfig(tuning_mode="deep_ptune", pre_seq_len=5)

    # Mocking nn.Module.register_parameter as instructed in memory
    _original_register_parameter = nn.Module.register_parameter

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    # The expected shape is (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (3, 2, 5, 16)
