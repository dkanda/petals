import sys
import os
sys.path.insert(0, os.path.abspath('src'))
import pytest
from unittest import mock

import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin, PTuneConfig
    from petals.utils.misc import DUMMY
except ImportError:
    pytest.skip('Skipping tests due to missing dependencies', allow_module_level=True)

class MockConfig(PretrainedConfig, PTuneConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 64
        self.num_hidden_layers = 4
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 10

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune():
    config = MockConfig()
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # num_hidden_layers is 4, so num_hidden_layers - 1 is 3
    assert intermediate_prompts.shape == (3, batch_size, config.pre_seq_len, config.hidden_size)

def test_ptune_only():
    config = MockConfig()
    config.tuning_mode = "ptune"
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts is DUMMY
