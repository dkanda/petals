import os
import sys

# Ensure local test environments without hivemind don't abort instantly
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

# We expect standard import to fail locally if hivemind is not installed,
# but they will pass in CI where dependencies are available.
try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Skipping test because petals/hivemind cannot be imported", allow_module_level=True)

class MockConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pre_seq_len = 5
        self.tuning_mode = "deep_ptune"
        self.hidden_size = 8
        self.num_hidden_layers = 4

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Mock word embeddings just for device and dtype access in get_prompt
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompt_shape():
    """
    Test that deep_ptune creates intermediate_prompts of the shape
    (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size).
    """
    config = MockConfig()
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Prompt shapes
    assert prompts.shape == torch.Size([batch_size, config.pre_seq_len, config.hidden_size])

    # intermediate_prompts should have shape (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    expected_shape = torch.Size([
        config.num_hidden_layers - 1,
        batch_size,
        config.pre_seq_len,
        config.hidden_size
    ])
    assert intermediate_prompts.shape == expected_shape
