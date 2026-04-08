import pytest
import sys
import torch
from unittest import mock

# Create mock package to avoid loading hivemind during testing
class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

sys.modules['hivemind'] = MockPackage()
sys.modules['petals'] = MockPackage()
sys.modules['petals.utils'] = MockPackage()
sys.modules['petals.utils.misc'] = type(sys)('petals.utils.misc')
sys.modules['petals.utils.misc'].DUMMY = torch.empty(0)

import importlib.util
spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
ptune_module = importlib.util.module_from_spec(spec)
sys.modules["petals.client.ptune"] = ptune_module
spec.loader.exec_module(ptune_module)

import torch.nn as nn
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tuning_mode = config.tuning_mode
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_mode():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="ptune",
        pre_seq_len=5
    )
    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert isinstance(intermediate_prompts, torch.Tensor) and intermediate_prompts.numel() == 0

def test_deep_ptune_mode():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=5
    )
    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")

    assert model.intermediate_prompt_embeddings.embedding_dim == (config.num_hidden_layers - 1) * config.hidden_size

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # Check that the first layer's prompts are all zeros
    assert torch.all(intermediate_prompts[0] == 0)

def test_invalid_tuning_mode():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="invalid_mode",
        pre_seq_len=5
    )
    with pytest.raises(NotImplementedError):
        MockModel(config)
