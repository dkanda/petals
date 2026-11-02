import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
    from petals.utils.misc import DUMMY
except ImportError:
    pytest.skip("Skipping test due to missing dependencies like hivemind", allow_module_level=True)

class DummyConfig(PretrainedConfig):
    def __init__(self, tuning_mode="deep_ptune", pre_seq_len=4, num_hidden_layers=3, hidden_size=8, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompt_shape():
    """Test that deep_ptune intermediate prompts have the expected shape (num_hidden_layers - 1, ...)."""
    config = DummyConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=4,
        num_hidden_layers=5,
        hidden_size=16
    )
    model = DummyModel(config)

    # Weight shape should be (pre_seq_len, (num_hidden_layers - 1) * hidden_size)
    assert model.intermediate_prompt_embeddings.weight.shape == (4, (5 - 1) * 16)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    # Shape should be (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    expected_shape = (5 - 1, batch_size, 4, 16)
    assert intermediate_prompts.shape == expected_shape

def test_ptune_normal_mode():
    """Test that normal ptune intermediate prompts is DUMMY."""
    config = DummyConfig(
        tuning_mode="ptune",
        pre_seq_len=4,
        num_hidden_layers=5,
        hidden_size=16
    )
    model = DummyModel(config)

    # intermediate_prompt_embeddings should not be created
    assert not hasattr(model, "intermediate_prompt_embeddings")

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert intermediate_prompts is DUMMY
