import os
import sys

# Force Petals dependencies to pass in sandbox
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

import pytest
import torch
from transformers import PretrainedConfig

try:
    import hivemind
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Missing dependencies for petals.client.ptune", allow_module_level=True)

class MockModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.init_prompts(config)

        # Mock word_embeddings
        self.word_embeddings = type('MockEmbeddings', (), {})()
        self.word_embeddings.weight = type('MockWeight', (), {})()
        self.word_embeddings.weight.device = torch.device("cpu")
        self.word_embeddings.weight.dtype = torch.float32

def test_deep_ptune_intermediate_prompt_shape():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=8,
        hidden_size=64,
        num_hidden_layers=12,
    )

    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 8, 64)
    # Expected shape: (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (11, 2, 8, 64)

def test_ptune_intermediate_prompt_shape():
    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=8,
        hidden_size=64,
        num_hidden_layers=12,
    )

    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 8, 64)
    assert intermediate_prompts.shape == (0,)
