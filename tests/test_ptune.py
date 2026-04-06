import sys
import unittest.mock as mock
import importlib.util
import pytest

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def test_deep_ptune():
    import torch

    mock_hivemind = MockPackage()
    mock_hivemind.get_logger = mock.MagicMock()
    mock_hivemind.utils = MockPackage()

    mock_sys_modules = {
        'hivemind': mock_hivemind,
        'hivemind.utils': mock_hivemind.utils,
        'petals': MockPackage(), # Avoid petals/__init__.py executing
        'petals.utils': MockPackage(),
        'petals.utils.misc': mock.MagicMock(),
    }

    with mock.patch.dict(sys.modules, mock_sys_modules):
        sys.modules['petals.utils.misc'].DUMMY = torch.zeros(1)

        spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
        ptune_module = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune_module
        spec.loader.exec_module(ptune_module)

        from transformers import PretrainedConfig

        class DummyModel(ptune_module.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = torch.nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=5,
            tuning_mode="deep_ptune",
            pre_seq_len=10,
        )
        model = DummyModel(config)

        # Assert correct parameters
        assert model.pre_seq_len == 10
        assert model.prompt_embeddings.weight.shape == (10, 64)
        # The fix should be num_hidden_layers - 1
        assert model.intermediate_prompt_embeddings.weight.shape == (10, (5 - 1) * 64)

        # Test get_prompt
        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        assert prompts.shape == (batch_size, 10, 64)
        assert intermediate_prompts.shape == (5, batch_size, 10, 64)

        # Assert the first layer is zero padded
        assert torch.all(intermediate_prompts[0] == 0)
