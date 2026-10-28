import sys
import os

import torch
import torch.nn as nn
from transformers import PretrainedConfig
import pytest

# We bypass the missing hivemind dependency for this test locally by wrapping it with try-except,
# but it will correctly run in CI where dependencies exist.
try:
    from petals.client.ptune import PTuneMixin
    from petals.utils.misc import DUMMY
except ImportError:
    pytest.skip("Skipping PTune test due to missing local dependency", allow_module_level=True)


class MockConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 10
        self.hidden_size = 64
        self.num_hidden_layers = 4


class MockModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)


def test_ptune_intermediate_prompt_embeddings_shape():
    config = MockConfig()
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
