import os
import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    import hivemind
    from petals.client.ptune import PTuneMixin
    from petals.utils.misc import DUMMY
except ImportError:
    pytest.skip("hivemind not available, skipping test", allow_module_level=True)

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_mixin_intermediate_prompts_shape():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=5
    )

    model = DummyModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (
        config.num_hidden_layers - 1,
        batch_size,
        config.pre_seq_len,
        config.hidden_size
    )

def test_ptune_mixin_ptune_mode():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="ptune",
        pre_seq_len=5
    )

    model = DummyModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts is DUMMY
