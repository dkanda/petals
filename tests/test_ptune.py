import sys
import pytest
import torch
import torch.nn as nn
from unittest import mock

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

mock_modules = {
    'hivemind': MockPackage(),
    'hivemind.utils': MockPackage(),
    'hivemind.utils.logging': MockPackage(),
    'transformers': MockPackage(),
    'petals': MockPackage(),
    'petals.utils': MockPackage(),
    'petals.utils.misc': MockPackage(),
}
patcher = mock.patch.dict('sys.modules', mock_modules)
patcher.start()

sys.path.insert(0, "src")
import importlib.util
spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
ptune = importlib.util.module_from_spec(spec)
sys.modules["ptune"] = ptune
spec.loader.exec_module(ptune)

PTuneMixin = ptune.PTuneMixin

class DummyConfig:
    def __init__(self, tuning_mode, pre_seq_len, hidden_size, num_hidden_layers):
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_logic():
    batch_size = 2
    pre_seq_len = 5
    hidden_size = 8
    num_hidden_layers = 4

    config = DummyConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=pre_seq_len,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers
    )

    model = DummyModel(config)

    # Check parameter count
    assert model.intermediate_prompt_embeddings.weight.shape == (pre_seq_len, (num_hidden_layers - 1) * hidden_size)

    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check shapes
    assert prompts.shape == (batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (num_hidden_layers, batch_size, pre_seq_len, hidden_size)

    # Verify zero padding on the first layer
    zero_padding = intermediate_prompts[0]
    assert torch.all(zero_padding == 0)

def test_ptune_logic():
    batch_size = 2
    pre_seq_len = 5
    hidden_size = 8
    num_hidden_layers = 4

    config = DummyConfig(
        tuning_mode="ptune",
        pre_seq_len=pre_seq_len,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers
    )

    model = DummyModel(config)

    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check shapes
    assert prompts.shape == (batch_size, pre_seq_len, hidden_size)

    # intermediate_prompts is DUMMY
    # In the mock, it returns DUMMY.to(dtype), which creates a new mock object.
    # So we check if the mock call signature implies it came from DUMMY
    assert "DUMMY.to()" in repr(intermediate_prompts) or intermediate_prompts is ptune.DUMMY
