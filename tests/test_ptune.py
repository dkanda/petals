import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin, PTuneConfig


class DummyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 64
        self.num_hidden_layers = 4
        self.pre_seq_len = 8
        self.tuning_mode = kwargs.get("tuning_mode", None)


class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)


def test_ptune_shapes():
    config = DummyConfig(tuning_mode="ptune")
    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts is None or intermediate_prompts.numel() == 0 # DUMMY


def test_deep_ptune_shapes():
    config = DummyConfig(tuning_mode="deep_ptune")
    model = DummyModel(config)

    batch_size = 3
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # Check intermediate_prompts shape: [num_hidden_layers, batch_size, pre_seq_len, hidden_size]
    assert intermediate_prompts.shape == (
        config.num_hidden_layers,
        batch_size,
        config.pre_seq_len,
        config.hidden_size,
    )

    # Check that the first layer prompt is purely zeros due to zero-padding
    assert torch.all(intermediate_prompts[0] == 0)
