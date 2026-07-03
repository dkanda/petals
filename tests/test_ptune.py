import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("hivemind is not available, skipping PTune tests", allow_module_level=True)


class DummyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = 4
        self.hidden_size = 16
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 5


class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)


def test_deep_ptune_intermediate_prompts_shape():
    """
    Test that intermediate_prompts have the correct shape in deep_ptune mode.
    The number of intermediate layers should be num_hidden_layers - 1.
    """
    config = DummyConfig()
    model = DummyModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    # prompts should be of shape [batch_size, pre_seq_len, hidden_size]
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # intermediate_prompts should be of shape [num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size]
    expected_intermediate_shape = (
        config.num_hidden_layers - 1,
        batch_size,
        config.pre_seq_len,
        config.hidden_size,
    )
    assert intermediate_prompts.shape == expected_intermediate_shape
