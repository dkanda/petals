import torch
import torch.nn as nn
from transformers import PretrainedConfig
import pytest

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_mixin_ptune_mode():
    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=3
    )

    model = DummyModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts is DUMMY

def test_ptune_mixin_deep_ptune_mode():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=3
    )

    model = DummyModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")

    # Check that intermediate prompt embeddings were initialized with correct shape
    expected_intermediate_dim = (config.num_hidden_layers - 1) * config.hidden_size
    assert model.intermediate_prompt_embeddings.weight.shape == (config.pre_seq_len, expected_intermediate_dim)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # The first layer should be zero-padded
    assert torch.all(intermediate_prompts[0] == 0)

    # Check intermediate prompts shapes are maintained even for a batch size of 1
    batch_size = 1
    prompts, intermediate_prompts = model.get_prompt(batch_size)
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)
    assert torch.all(intermediate_prompts[0] == 0)
