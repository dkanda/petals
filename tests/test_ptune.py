import os
import sys

import torch
import pytest

try:
    import hivemind
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip('hivemind is not available', allow_module_level=True)

from transformers import PretrainedConfig

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        class MockWordEmbeddings:
            weight = torch.empty((1, 1), dtype=torch.float32)
        self.word_embeddings = MockWordEmbeddings()
        self.init_prompts(config)

def test_ptune_intermediate_prompt_shape():
    """
    Test that intermediate prompts in deep_ptune mode are correctly shaped
    for (num_hidden_layers - 1) layers.
    """
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=8,
        num_hidden_layers=3
    )
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 8)
    # the expected shape is (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (2, 2, 5, 8)
