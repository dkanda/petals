import sys
import unittest.mock

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class MockModelConfig(PretrainedConfig):
    def __init__(self, hidden_size=64, num_hidden_layers=4, pre_seq_len=10, tuning_mode=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.pre_seq_len = pre_seq_len
        self.tuning_mode = tuning_mode

class MockModelWithPTune(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    config = MockModelConfig(tuning_mode="ptune")
    model = MockModelWithPTune(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts is DUMMY

def test_deep_ptune_shapes():
    config = MockModelConfig(tuning_mode="deep_ptune")
    model = MockModelWithPTune(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")

    # Check that intermediate prompt embeddings handle layers - 1
    expected_intermediate_shape = (config.num_hidden_layers - 1) * config.hidden_size
    assert model.intermediate_prompt_embeddings.weight.shape == (config.pre_seq_len, expected_intermediate_shape)

    batch_size = 3
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    # The intermediate prompts must be correctly zero-padded at the first layer, thus total layers = num_hidden_layers
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # Ensure the first layer's padding is entirely zeros
    assert torch.all(intermediate_prompts[0] == 0)

if __name__ == "__main__":
    pytest.main([__file__])