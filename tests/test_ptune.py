import os
import sys

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.utils.misc import force_non_empty_weights

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Skipping PTuneMixin tests due to missing dependencies", allow_module_level=True)


class DummyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 10
        self.hidden_size = 16
        self.num_hidden_layers = 4


class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)


def test_ptune_mixin_intermediate_prompts_shape():
    config = DummyConfig()
    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (
        config.num_hidden_layers - 1,
        batch_size,
        config.pre_seq_len,
        config.hidden_size,
    )
