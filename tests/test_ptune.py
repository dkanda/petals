import sys
import os

# Add src to the Python path
sys.path.insert(0, os.path.abspath('src'))

import pytest
import torch
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip('Skipping test due to missing dependencies like hivemind', allow_module_level=True)

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.init_prompts(config)
        self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)

def test_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        num_hidden_layers=4,
        hidden_size=8
    )

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 8)
    assert intermediate_prompts.shape == (3, 2, 5, 8)
