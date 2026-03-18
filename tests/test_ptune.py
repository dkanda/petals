import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class DummyConfig(PretrainedConfig):
    def __init__(self, tuning_mode=None, pre_seq_len=0, hidden_size=64, num_hidden_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_mixin_ptune():
    config = DummyConfig(tuning_mode="ptune", pre_seq_len=5, hidden_size=16)
    model = DummyModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")
    assert model.prompt_embeddings.weight.shape == (5, 16)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts is DUMMY

def test_ptune_mixin_deep_ptune():
    config = DummyConfig(tuning_mode="deep_ptune", pre_seq_len=7, hidden_size=16, num_hidden_layers=4)
    model = DummyModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")

    # intermediate_prompt_embeddings should have shape (pre_seq_len, (num_hidden_layers - 1) * hidden_size)
    assert model.intermediate_prompt_embeddings.weight.shape == (7, (4 - 1) * 16)

    prompts, intermediate_prompts = model.get_prompt(batch_size=3)

    # prompts should be (batch_size, pre_seq_len, hidden_size)
    assert prompts.shape == (3, 7, 16)

    # intermediate_prompts should be (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (4, 3, 7, 16)

    # The first layer of intermediate_prompts should be all zeros
    assert torch.all(intermediate_prompts[0] == 0)

    # The rest should not necessarily be zeros (they are initialized with random weights)
    # We just check that the sum of the absolute values is greater than 0
    assert torch.sum(torch.abs(intermediate_prompts[1:])) > 0
