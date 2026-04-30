import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin

class MockPretrainedConfig(PretrainedConfig):
    def __init__(self, num_hidden_layers=3, hidden_size=4, pre_seq_len=2, tuning_mode="deep_ptune", **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.pre_seq_len = pre_seq_len
        self.tuning_mode = tuning_mode

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune():
    config = MockPretrainedConfig(num_hidden_layers=3, hidden_size=4, pre_seq_len=2, tuning_mode="deep_ptune")
    model = DummyModel(config)

    assert model.intermediate_prompt_embeddings.weight.shape == (2, 8)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 2, 4)
    # The new shape should match num_hidden_layers=3
    assert intermediate_prompts.shape == (3, 2, 2, 4)
    # Ensure the first layer prompts are all zeroes (padding layer)
    assert torch.all(intermediate_prompts[0] == 0)
