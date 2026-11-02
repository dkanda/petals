import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Skipping test because petals/hivemind is missing", allow_module_level=True)

class MockConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 8
        self.num_hidden_layers = 4
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 5

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = MockConfig()
    model = MockModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # prompts should be [batch_size, pre_seq_len, hidden_size]
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # intermediate_prompts should be [num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size]
    expected_intermediate_shape = (
        config.num_hidden_layers - 1,
        batch_size,
        config.pre_seq_len,
        config.hidden_size
    )
    assert intermediate_prompts.shape == expected_intermediate_shape
