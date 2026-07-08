import os
import sys

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

sys.path.insert(0, os.path.abspath("src"))

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("hivemind not installed", allow_module_level=True)


class DummyConfig(PretrainedConfig):
    def __init__(self):
        super().__init__()
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 5
        self.num_hidden_layers = 10
        self.hidden_size = 32


class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = nn.Embedding(10, 32)
        self.init_prompts(config)


def test_ptune_intermediate_prompts_shape():
    model = DummyModel(DummyConfig())
    # 5, (10 - 1) * 32 = 5, 288
    assert model.intermediate_prompt_embeddings.weight.shape == (5, 288)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    # 9, 2, 5, 32
    assert intermediate_prompts.shape == (9, 2, 5, 32)
