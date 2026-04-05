import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import importlib.util
from unittest.mock import MagicMock, patch
import sys
import builtins

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

def get_ptune_module():
    # Setup mocks
    mocks = {
        'hivemind': MockPackage(),
        'hivemind.utils': MockPackage(),
        'hivemind.moe': MockPackage(),
        'hivemind.moe.client': MockPackage(),
        'hivemind.moe.client.remote_expert_worker': MockPackage(),
        'hivemind.p2p': MockPackage(),
        'petals.utils.misc': MagicMock()
    }
    mocks['petals.utils.misc'].DUMMY = torch.zeros(1)

    original_import = builtins.__import__
    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("petals.") and name != "petals.utils.misc":
            return MockPackage()
        return original_import(name, globals, locals, fromlist, level)

    with patch.dict(sys.modules, mocks), patch('builtins.__import__', mock_import):
        spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)
        return ptune

ptune = get_ptune_module()
PTuneMixin = ptune.PTuneMixin

class MockConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = kwargs.get("hidden_size", 64)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 4)
        self.pre_seq_len = kwargs.get("pre_seq_len", 8)
        self.tuning_mode = kwargs.get("tuning_mode", None)

class MockModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_initialization():
    config = MockConfig(tuning_mode="deep_ptune", num_hidden_layers=5, hidden_size=16, pre_seq_len=10)
    model = MockModel(config)

    assert hasattr(model, 'intermediate_prompt_embeddings'), "intermediate_prompt_embeddings missing"

    # Expected size should be (config.num_hidden_layers - 1) * config.hidden_size
    expected_size = (5 - 1) * 16
    assert model.intermediate_prompt_embeddings.weight.shape == (10, expected_size), "Incorrect intermediate prompt embeddings shape"

def test_deep_ptune_get_prompt():
    config = MockConfig(tuning_mode="deep_ptune", num_hidden_layers=3, hidden_size=8, pre_seq_len=4)
    model = MockModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Expected shapes
    # prompts: (batch_size, pre_seq_len, hidden_size)
    # intermediate_prompts: (num_hidden_layers, batch_size, pre_seq_len, hidden_size)

    assert prompts.shape == (2, 4, 8)
    assert intermediate_prompts.shape == (3, 2, 4, 8)

    # Verify the first layer is padded with zeros
    zero_padding = intermediate_prompts[0]
    assert torch.all(zero_padding == 0), "The first block prompt was not padded with zeros"
