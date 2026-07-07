import os
import sys
import pytest

try:
    from transformers import PretrainedConfig
    import torch
    import torch.nn as nn
    from petals.client.ptune import PTuneMixin
except ImportError as e:
    pytest.skip(str(e), allow_module_level=True)

def test_ptune_intermediate_prompts():
    class DummyConfig(PretrainedConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = 8
            self.num_hidden_layers = 4
            self.tuning_mode = "deep_ptune"
            self.pre_seq_len = 3

    class DummyModel(PTuneMixin, nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

    config = DummyConfig()
    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
