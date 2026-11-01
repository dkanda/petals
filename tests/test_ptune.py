import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import os
import sys

try:
    import hivemind.moe
    from hivemind import PeerID
    from petals.client.ptune import PTuneMixin, PTuneConfig
except ImportError as e:
    pytest.skip(f"petals or dependencies not available: {e}", allow_module_level=True)


class DummyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pre_seq_len = 5
        self.tuning_mode = "deep_ptune"
        self.hidden_size = 8
        self.num_hidden_layers = 4

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    config = DummyConfig()
    model = DummyModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == torch.Size([2, 5, 8])
    assert intermediate_prompts.shape == torch.Size([3, 2, 5, 8])
