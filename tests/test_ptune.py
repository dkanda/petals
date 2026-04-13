import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=4,
        pre_seq_len=5,
        tuning_mode="deep_ptune",
    )
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)
    assert (intermediate_prompts[0] == 0).all() # First layer should be zero-padded
