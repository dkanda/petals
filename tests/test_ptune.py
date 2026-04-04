import pytest
import sys
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from unittest import mock
import importlib.util

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

# Mock hivemind for the direct module load
sys.modules['hivemind'] = MockPackage()
sys.modules['hivemind.get_logger'] = mock.MagicMock()

# Load the module dynamically to avoid triggering the whole petals/__init__.py imports
spec = importlib.util.spec_from_file_location("ptune_module", "src/petals/client/ptune.py")
ptune_module = importlib.util.module_from_spec(spec)
sys.modules["ptune_module"] = ptune_module

# Need to mock petals.utils.misc as well for the direct load
sys.modules['petals'] = MockPackage()
sys.modules['petals.utils'] = MockPackage()
sys.modules['petals.utils.misc'] = MockPackage()
sys.modules['petals.utils.misc'].DUMMY = torch.zeros(1)

spec.loader.exec_module(ptune_module)

PTuneMixin = ptune_module.PTuneMixin
DUMMY = ptune_module.DUMMY

class MockWordEmbeddings:
    def __init__(self, hidden_size):
        self.weight = nn.Parameter(torch.randn(10, hidden_size))

class MockConfig(PretrainedConfig):
    def __init__(self, tuning_mode=None, pre_seq_len=0, hidden_size=64, num_hidden_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = MockWordEmbeddings(config.hidden_size)
        self.init_prompts(config)

def test_ptune_initialization_and_forward():
    config = MockConfig(tuning_mode="ptune", pre_seq_len=5, hidden_size=16)
    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert "DUMMY" in repr(intermediate_prompts) or intermediate_prompts is DUMMY or torch.all(intermediate_prompts == 0)
    assert prompts.dtype == model.word_embeddings.weight.dtype

def test_deep_ptune_initialization_and_forward():
    config = MockConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=4)
    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")

    # Check intermediate embeddings dimension (num_hidden_layers - 1)
    expected_intermediate_dim = (config.num_hidden_layers - 1) * config.hidden_size
    assert model.intermediate_prompt_embeddings.weight.shape[1] == expected_intermediate_dim

    batch_size = 3
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # prompts should be (batch_size, pre_seq_len, hidden_size)
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # intermediate_prompts should be padded with zeros at dim=0 to length num_hidden_layers
    # shape: (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # Verify the first layer's intermediate prompts are all zeros
    assert torch.all(intermediate_prompts[0] == 0)

    # Verify dtypes match
    assert prompts.dtype == model.word_embeddings.weight.dtype
    assert intermediate_prompts.dtype == model.word_embeddings.weight.dtype
