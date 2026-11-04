import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
    from petals.utils.misc import DUMMY
except ImportError:
    pytest.skip("Skipping test due to missing dependencies like hivemind in isolated testing environment", allow_module_level=True)

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="ptune",
        pre_seq_len=5
    )

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(2)

    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts is DUMMY

def test_deep_ptune_shapes():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=5
    )

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(2)

    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (3, 2, 5, 16)
