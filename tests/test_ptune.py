import pytest
import os

try:
    import hivemind
except ImportError:
    pytest.skip("hivemind not available", allow_module_level=True)

from petals.client.ptune import PTuneMixin
import torch
import torch.nn as nn
from transformers import PretrainedConfig

class MockModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompt_shape():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )
    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (3, 2, 5, 16)
