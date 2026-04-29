import pytest
import torch
import unittest.mock as mock
from transformers import PretrainedConfig
import sys

def get_ptune_with_mocks():
    """Returns the ptune module with mocked dependencies to avoid import errors in isolated tests."""
    class MockPackage(mock.MagicMock):
        __path__ = []

    mock_modules = {
        'hivemind': MockPackage(),
        'petals': MockPackage(),
        'petals.utils.misc': mock.MagicMock(DUMMY=torch.zeros(1))
    }

    import importlib.util
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)

    with mock.patch.dict('sys.modules', mock_modules):
        spec.loader.exec_module(ptune)

    return ptune

def test_ptune_shapes():
    with mock.patch.dict('sys.modules', {
        'hivemind': mock.MagicMock(__path__=[]),
        'petals': mock.MagicMock(__path__=[]),
        'petals.utils.misc': mock.MagicMock(DUMMY=torch.zeros(1))
    }):
        ptune = get_ptune_with_mocks()

        class MockModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = type('Mock', (), {'weight': torch.empty(0)})()
                self.init_prompts(self.config)

        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=10
        )
        config.tuning_mode = "ptune"
        config.pre_seq_len = 5

        model = MockModel(config)
        p, ip = model.get_prompt(batch_size=3)
        assert p.shape == (3, 5, 64)
        assert ip.numel() == 1 # Since DUMMY is torch.zeros(1)

def test_deep_ptune_shapes():
    with mock.patch.dict('sys.modules', {
        'hivemind': mock.MagicMock(__path__=[]),
        'petals': mock.MagicMock(__path__=[]),
        'petals.utils.misc': mock.MagicMock(DUMMY=torch.zeros(1))
    }):
        ptune = get_ptune_with_mocks()

        class MockModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = type('Mock', (), {'weight': torch.empty(0)})()
                self.init_prompts(self.config)

        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=10
        )
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 5

        model = MockModel(config)
        p, ip = model.get_prompt(batch_size=3)

        assert p.shape == (3, 5, 64)
        assert ip.shape == (10, 3, 5, 64)

        # Check that the first layer (dim=0 index 0) of ip is zero-padded
        assert torch.all(ip[0] == 0)
