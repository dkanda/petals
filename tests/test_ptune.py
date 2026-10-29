import sys
import os
import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass

try:
    from petals.client.ptune import PTuneMixin
except ImportError as e:
    pytest.skip(f"Skipping test due to missing dependency: {e}", allow_module_level=True)

@dataclass
class DummyConfig:
    tuning_mode: str = "deep_ptune"
    pre_seq_len: int = 5
    hidden_size: int = 32
    num_hidden_layers: int = 4

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = DummyConfig()
    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == torch.Size([batch_size, config.pre_seq_len, config.hidden_size])
    assert intermediate_prompts.shape == torch.Size([config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size])
