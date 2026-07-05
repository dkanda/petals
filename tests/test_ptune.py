import os
import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
    from petals.utils.misc import DUMMY
except ImportError:
    pytest.skip("hivemind not available, skipping test locally", allow_module_level=True)

class MockModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=10,
    )
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5

    model = MockModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts is not DUMMY
    assert intermediate_prompts.shape == (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)

def test_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=10,
    )
    config.tuning_mode = "ptune"
    config.pre_seq_len = 5

    model = MockModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts is DUMMY
