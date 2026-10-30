import os
import sys
import pytest

import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin, PTuneConfig
except ImportError:
    pytest.skip("Skipping because hivemind or petals is not fully available locally", allow_module_level=True)

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=12,
        tuning_mode="deep_ptune",
        pre_seq_len=8,
    )
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 8, 64)
    # The intermediate prompts should have shape (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (11, 2, 8, 64)
