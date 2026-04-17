import pytest
import torch
import torch.nn as nn
import unittest.mock as mock
import importlib.util
import sys

def test_ptune_isolated():
    """
    Test PTuneMixin in an isolated environment by mocking sys.modules directly
    inside the test function to prevent global scope poisoning that breaks other tests.
    """
    class MockPackage(mock.MagicMock):
        __path__ = []
        __spec__ = None

    # Patch sys.modules safely using a context manager
    with mock.patch.dict(
        "sys.modules",
        {
            "hivemind": MockPackage(),
            "hivemind.utils": MockPackage(),
            "petals": MockPackage(),
            "petals.utils": MockPackage(),
            "petals.utils.misc": MockPackage(),
        }
    ):
        sys.modules["petals.utils.misc"].DUMMY = torch.empty(0)

        # Now import locally
        spec = importlib.util.spec_from_file_location("ptune_isolated", "src/petals/client/ptune.py")
        ptune_module = importlib.util.module_from_spec(spec)
        with mock.patch.dict("sys.modules", {"torch.nn.Module.register_parameter": mock.MagicMock()}):
            spec.loader.exec_module(ptune_module)

        class MockConfig:
            def __init__(self, tuning_mode, pre_seq_len, hidden_size, num_hidden_layers):
                self.tuning_mode = tuning_mode
                self.pre_seq_len = pre_seq_len
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers

        class TestPTuneMixin(ptune_module.PTuneMixin):
            def __init__(self, config):
                self.config = config
                # Mock word embeddings normally accessed via self.word_embeddings.weight
                self.word_embeddings = type('MockWordEmbeddings', (), {'weight': torch.zeros(1)})()

        # Test init_prompts
        config = MockConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=10, num_hidden_layers=4)
        model = TestPTuneMixin(config)

        with mock.patch.object(torch.nn.Module, "register_parameter", new=mock.MagicMock()):
            model.init_prompts(config)

        assert model.intermediate_prompt_embeddings.weight.shape == (5, (4 - 1) * 10)

        # Test get_prompt
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert prompts.shape == (2, 5, 10)
        assert intermediate_prompts.shape == (4, 2, 5, 10)

        # Assert zero padding on the first layer
        assert torch.all(intermediate_prompts[0] == 0)
