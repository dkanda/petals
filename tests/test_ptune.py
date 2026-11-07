import sys
import os

sys.path.insert(0, os.path.abspath('src'))

import torch
import torch.nn as nn
from transformers import PretrainedConfig
import pytest

try:
    from hivemind import PeerID
    from petals.client.ptune import PTuneMixin
    from petals.utils.misc import DUMMY
except ImportError:
    pytest.skip('hivemind not installed', allow_module_level=True)

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompt_shape():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=3
    )
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(2)
    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (2, 2, 5, 16)
