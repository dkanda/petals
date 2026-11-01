import os
import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    import hivemind
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("hivemind is not available in the local environment", allow_module_level=True)


class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)


def test_ptune_intermediate_prompt_shape():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        num_hidden_layers=4,
        hidden_size=8,
    )
    model = DummyModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    expected_intermediate_shape = (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == expected_intermediate_shape, \
        f"Expected {expected_intermediate_shape}, got {intermediate_prompts.shape}"
