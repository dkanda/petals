import pytest
import torch
import torch.nn as nn
from unittest import mock
import sys

# We will need to bypass petals imports if we run isolated tests
from petals.client.ptune import PTuneMixin, force_non_empty_weights
from transformers import PretrainedConfig

class MockModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)

        with mock.patch('petals.client.ptune._original_register_parameter', torch.nn.Module.register_parameter):
            self.init_prompts(config)

def test_deep_ptune():
    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5
    config.hidden_size = 16
    config.num_hidden_layers = 4

    model = MockModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert intermediate_prompts.shape[0] == config.num_hidden_layers - 1
