import pytest
import os
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import sys
import unittest.mock

try:
    import hivemind
    os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Missing dependencies", allow_module_level=True)

class MockModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=8
    )

    model = MockModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (
        config.num_hidden_layers - 1,
        batch_size,
        config.pre_seq_len,
        config.hidden_size
    )
