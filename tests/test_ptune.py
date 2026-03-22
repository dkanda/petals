import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin
from petals.utils.misc import is_dummy

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
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    config = MockConfig(tuning_mode="ptune", pre_seq_len=5, hidden_size=64, num_hidden_layers=4)
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check shapes for regular ptune
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert is_dummy(intermediate_prompts)

def test_deep_ptune_shapes():
    config = MockConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=64, num_hidden_layers=4)
    model = MockModel(config)

    # Check intermediate embeddings weight shape (should be num_hidden_layers - 1)
    assert model.intermediate_prompt_embeddings.weight.shape == (
        config.pre_seq_len,
        (config.num_hidden_layers - 1) * config.hidden_size,
    )

    batch_size = 3
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check shape of main prompts (used for the first layer)
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # Check shape of intermediate prompts (should include all layers, first layer is zero padded)
    assert intermediate_prompts.shape == (
        config.num_hidden_layers,
        batch_size,
        config.pre_seq_len,
        config.hidden_size,
    )

    # Verify the first layer is exactly zero padding
    first_layer_prompts = intermediate_prompts[0]
    assert torch.all(first_layer_prompts == 0)

def test_ptune_invalid_mode():
    config = MockConfig(tuning_mode="invalid_mode", pre_seq_len=5)
    with pytest.raises(NotImplementedError):
        MockModel(config)

def test_ptune_zero_seq_len():
    config = MockConfig(tuning_mode="ptune", pre_seq_len=0)
    with pytest.raises(AssertionError):
        MockModel(config)
