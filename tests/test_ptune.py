import os
import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
except ImportError as e:
    pytest.skip(f"Skipping test due to missing dependencies: {e}", allow_module_level=True)


class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)


def test_deep_ptune_shapes():
    """
    Test that deep_ptune correctly sets intermediate_prompt_embeddings to have a shape that
    corresponds to num_hidden_layers - 1.
    """
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=3,
        pre_seq_len=4,
        tuning_mode="deep_ptune"
    )

    model = DummyModel(config)

    # Weight shape should be pre_seq_len x ((num_hidden_layers - 1) * hidden_size)
    assert model.intermediate_prompt_embeddings.weight.shape == (4, 128)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    # Prompts shape should be (batch_size, pre_seq_len, hidden_size)
    assert prompts.shape == (2, 4, 64)

    # Intermediate prompts shape should be (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (2, 2, 4, 64)
