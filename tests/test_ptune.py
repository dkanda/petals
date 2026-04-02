import torch
import torch.nn as nn
from transformers import PretrainedConfig
import pytest
import sys
from unittest.mock import MagicMock

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

import os

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class MockConfig(PretrainedConfig):
    def __init__(self, hidden_size=64, num_hidden_layers=4, tuning_mode=None, pre_seq_len=0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len

class MockModelWithPTune(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)

        # Calling the mixin init
        self.init_prompts(config)

def test_ptune_disabled():
    config = MockConfig()
    model = MockModelWithPTune(config)
    assert not hasattr(model, 'prompt_embeddings')
    assert not hasattr(model, 'intermediate_prompt_embeddings')

def test_ptune_enabled():
    config = MockConfig(tuning_mode="ptune", pre_seq_len=10)
    model = MockModelWithPTune(config)

    assert hasattr(model, 'prompt_embeddings')
    assert model.prompt_embeddings.weight.shape == (10, 64)
    assert not hasattr(model, 'intermediate_prompt_embeddings')

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)
    assert prompts.shape == (batch_size, 10, 64)
    assert intermediate_prompts is DUMMY

def test_deep_ptune_enabled():
    config = MockConfig(tuning_mode="deep_ptune", pre_seq_len=10)
    model = MockModelWithPTune(config)

    assert hasattr(model, 'prompt_embeddings')
    assert hasattr(model, 'intermediate_prompt_embeddings')

    # Check dimensions: should be (num_hidden_layers - 1) * hidden_size
    expected_dim = (config.num_hidden_layers - 1) * config.hidden_size
    assert model.intermediate_prompt_embeddings.weight.shape == (10, expected_dim)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, 10, 64)

    # intermediate_prompts should have prepended zero padding for layer 0
    # Expected shape: [num_hidden_layers, batch_size, pre_seq_len, hidden_size]
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, 10, 64)

    # Check that the first layer in intermediate_prompts is all zeros
    assert torch.all(intermediate_prompts[0] == 0)

    # Check that subsequent layers are not necessarily all zeros (from embedding layer initialization)
    assert not torch.all(intermediate_prompts[1:] == 0)

if __name__ == "__main__":
    pytest.main([__file__])
