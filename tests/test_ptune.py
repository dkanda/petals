import os
import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    import hivemind
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("hivemind is not available. Skipping test.", allow_module_level=True)


class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)


def test_deep_ptune_intermediate_prompt_shape():
    """
    Test that when using deep_ptune, the intermediate prompts have shape:
    (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    """
    config = PretrainedConfig(
        tuning_mode='deep_ptune',
        pre_seq_len=5,
        hidden_size=64,
        num_hidden_layers=3
    )

    model = DummyModel(config)
    model.init_prompts(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    expected_shape = (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == expected_shape, f"Expected shape {expected_shape}, got {intermediate_prompts.shape}"
