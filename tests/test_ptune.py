import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class DummyConfig(PretrainedConfig):
    def __init__(self, tuning_mode=None, pre_seq_len=0, num_hidden_layers=10, hidden_size=32, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_mode():
    config = DummyConfig(tuning_mode="ptune", pre_seq_len=5, num_hidden_layers=10, hidden_size=32)
    model = DummyModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")
    assert model.prompt_embeddings.weight.shape == (5, 32)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (2, 5, 32)
    assert intermediate_prompts is DUMMY

def test_deep_ptune_mode():
    config = DummyConfig(tuning_mode="deep_ptune", pre_seq_len=5, num_hidden_layers=10, hidden_size=32)
    model = DummyModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")
    assert model.prompt_embeddings.weight.shape == (5, 32)
    assert model.intermediate_prompt_embeddings.weight.shape == (5, 9 * 32)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (2, 5, 32)
    # Shape should be (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (10, 2, 5, 32)

    # Check that the first layer's prompts in intermediate_prompts are all zeros
    assert torch.all(intermediate_prompts[0] == 0)
