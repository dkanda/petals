import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

import sys
sys.path.insert(0, 'src')
import os
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

try:
    from hivemind import PeerID
    from petals.client.ptune import PTuneMixin
except ImportError as e:
    pytest.skip(f"Could not import petals due to missing dependencies: {e}", allow_module_level=True)

class MockConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 5
        self.hidden_size = 16
        self.num_hidden_layers = 4

class MockModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_mixin_intermediate_prompts_shape():
    config = MockConfig()
    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (3, 2, 5, 16) # (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
