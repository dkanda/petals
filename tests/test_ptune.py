import sys
import unittest.mock as mock

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

sys.modules['hivemind'] = MockPackage()
sys.modules['petals'] = MockPackage()
sys.modules['petals.utils'] = MockPackage()
sys.modules['petals.utils.misc'] = MockPackage()

import importlib.util
spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
ptune_module = importlib.util.module_from_spec(spec)

import torch
import torch.nn as nn
from transformers import PretrainedConfig

class DummyTensor:
    def to(self, *args, **kwargs):
        return self
    def __repr__(self):
        return "DUMMY.to()"
DUMMY = DummyTensor()
sys.modules['petals.utils.misc'].DUMMY = DUMMY

ptune_module.torch = torch
ptune_module.nn = nn
ptune_module.PretrainedConfig = PretrainedConfig
ptune_module.DUMMY = DUMMY
ptune_module.get_logger = mock.MagicMock()

spec.loader.exec_module(ptune_module)

import pytest

class MockConfig(PretrainedConfig):
    def __init__(self, tuning_mode=None, pre_seq_len=0, hidden_size=64, num_hidden_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class MockModel(ptune_module.PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_mixin_deep_ptune():
    batch_size = 2
    pre_seq_len = 5
    hidden_size = 16
    num_hidden_layers = 3

    config = MockConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=pre_seq_len,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers
    )

    model = MockModel(config)

    # Check that intermediate embeddings have size for (num_hidden_layers - 1)
    assert model.intermediate_prompt_embeddings.weight.shape == (pre_seq_len, (num_hidden_layers - 1) * hidden_size)

    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, pre_seq_len, hidden_size)

    # Check intermediate prompts have been padded to length num_hidden_layers
    assert intermediate_prompts.shape == (num_hidden_layers, batch_size, pre_seq_len, hidden_size)

    # Check that the first layer (index 0) of intermediate_prompts is all zeros
    assert torch.all(intermediate_prompts[0] == 0)

def test_ptune_mixin_ptune():
    batch_size = 2
    pre_seq_len = 5
    hidden_size = 16
    num_hidden_layers = 3

    config = MockConfig(
        tuning_mode="ptune",
        pre_seq_len=pre_seq_len,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers
    )

    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")

    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, pre_seq_len, hidden_size)
    assert "DUMMY.to()" in repr(intermediate_prompts) or intermediate_prompts is DUMMY
