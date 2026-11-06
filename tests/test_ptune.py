import os
import sys

# Ensure local imports can be resolved
sys.path.insert(0, os.path.abspath("src"))

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    import hivemind
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Skipping local tests due to missing dependencies like hivemind.", allow_module_level=True)

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_shapes():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=3
    )
    model = DummyModel(config)
    prompts, intermediate = model.get_prompt(batch_size=2)

    assert prompts.shape == torch.Size([2, 5, 16])
    assert intermediate.shape == torch.Size([2, 2, 5, 16])
