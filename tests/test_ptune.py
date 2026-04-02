import unittest.mock
import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import sys

# Mocking hivemind to avoid ImportErrors
class MockPackage:
    __path__ = []
    __spec__ = None

    def __getattr__(self, name):
        if name in ["get_logger"]:
            return lambda *args, **kwargs: unittest.mock.MagicMock()
        return unittest.mock.MagicMock()

def test_deep_ptune():
    mock_modules = {
        'hivemind': MockPackage(),
        'hivemind.get_logger': MockPackage(),
        'petals': MockPackage(), # Mock the root package to prevent __init__.py evaluation
        'petals.utils': MockPackage(),
        'petals.utils.misc': MockPackage(),
    }

    with unittest.mock.patch.dict('sys.modules', mock_modules):
        # Provide a dummy DUMMY
        import sys
        sys.modules['petals.utils.misc'].DUMMY = torch.empty(0)

        import importlib.util
        # Dynamically import PTuneMixin avoiding petals.__init__.py
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune_module = importlib.util.module_from_spec(spec)
        sys.modules["ptune"] = ptune_module
        spec.loader.exec_module(ptune_module)
        PTuneMixin = ptune_module.PTuneMixin

        class MockModel(PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)

        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=4,
            tuning_mode="deep_ptune",
            pre_seq_len=5
        )

        model = MockModel(config)
        model.init_prompts(config)

        # Verify intermediate_prompt_embeddings dimensions
        # Should have parameters for num_hidden_layers - 1
        assert model.intermediate_prompt_embeddings.weight.shape == (
            config.pre_seq_len,
            (config.num_hidden_layers - 1) * config.hidden_size
        )

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        # Verify the shapes
        assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

        # intermediate_prompts should have prepended zero-padding for the first layer
        # Its shape should be (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
        assert intermediate_prompts.shape == (
            config.num_hidden_layers,
            batch_size,
            config.pre_seq_len,
            config.hidden_size
        )

        # Verify zero-padding in the first layer
        assert torch.all(intermediate_prompts[0] == 0)

        # Verify other layers are not entirely zero (unless they randomly initialized to 0, which is extremely unlikely)
        assert not torch.all(intermediate_prompts[1:] == 0)
