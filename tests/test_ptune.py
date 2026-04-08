import pytest
import torch
from transformers import PretrainedConfig
from typing import Optional
from unittest import mock
import sys
import importlib.util

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

class DummyConfig(PretrainedConfig):
    def __init__(self, pre_seq_len=10, tuning_mode=None, hidden_size=64, num_hidden_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.pre_seq_len = pre_seq_len
        self.tuning_mode = tuning_mode
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

def test_deep_ptune():
    mock_modules = {
        'hivemind': MockPackage(),
        'hivemind.utils': MockPackage(),
        'hivemind.utils.logging': MockPackage(),
        'hivemind.moe': MockPackage(),
        'hivemind.moe.client': MockPackage(),
        'hivemind.moe.client.remote_expert_worker': MockPackage(),
        'hivemind.p2p': MockPackage(),
        'petals.utils.misc': mock.MagicMock(),
    }
    mock_modules['petals.utils.misc'].DUMMY = torch.empty(0)

    with mock.patch.dict('sys.modules', mock_modules):
        # To avoid full petals loading, we load just the file we need dynamically
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["ptune"] = ptune
        spec.loader.exec_module(ptune)

        class DummyModel(ptune.PTuneMixin, torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = torch.nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        config = DummyConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=3)
        model = DummyModel(config)

        assert hasattr(model, 'prompt_embeddings')
        assert hasattr(model, 'intermediate_prompt_embeddings')

        assert model.intermediate_prompt_embeddings.weight.shape == (5, (3 - 1) * 16)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert prompts.shape == (2, 5, 16)

        # We should have shape (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
        assert intermediate_prompts.shape == (3, 2, 5, 16)

        # Check that the first layer prompt is correctly zero-padded
        assert torch.all(intermediate_prompts[0] == 0)
