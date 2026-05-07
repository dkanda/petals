import pytest
import unittest.mock as mock
import torch
import torch.nn as nn

def test_ptune_mixin_deep_ptune():
    # Setup mocks to allow importing the module directly
    import sys
    import types

    mock_misc = mock.Mock()
    mock_misc.DUMMY = torch.empty(0)

    mock_hivemind = types.ModuleType('hivemind')
    mock_hivemind.__path__ = []
    mock_hivemind.get_logger = mock.Mock()

    mock_petals = types.ModuleType('petals')
    mock_petals.__path__ = []
    mock_petals.client = types.ModuleType('petals.client')
    mock_petals.client.__path__ = []

    with mock.patch.dict('sys.modules', {
        'petals': mock_petals,
        'petals.client': mock_petals.client,
        'petals.utils.misc': mock_misc,
        'hivemind': mock_hivemind
    }):
        import importlib.util
        spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

        class DummyConfig:
            def __init__(self):
                self.tuning_mode = "deep_ptune"
                self.pre_seq_len = 4
                self.hidden_size = 8
                self.num_hidden_layers = 3

        class DummyModel(nn.Module, ptune.PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)

        config = DummyConfig()
        model = DummyModel(config)

        # Test initialization
        model.init_prompts(config)
        assert model.intermediate_prompt_embeddings.weight.shape == (4, (3 - 1) * 8)

        # Test get_prompt
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert intermediate_prompts.shape == (3 - 1, 2, 4, 8)
