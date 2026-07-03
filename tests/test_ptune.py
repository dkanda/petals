import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import os

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Skipping due to missing petals dependency", allow_module_level=True)

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_shapes():
    config = PretrainedConfig(hidden_size=16, num_hidden_layers=4)
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5

    model = MockModel(config)
    prompts, intermediate = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 16)
    assert intermediate.shape == (3, 2, 5, 16)
