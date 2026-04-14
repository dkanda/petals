import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from unittest import mock
import sys

# Define MockPackage globally
class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def test_ptune_mode():
    # Only mock for this test execution by patching sys.modules
    mock_modules = {
        'hivemind': MockPackage(),
        'petals': MockPackage(),
    }

    with mock.patch.dict('sys.modules', mock_modules):
        # Dynamically load the module directly to avoid standard __init__.py issues
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune_module = importlib.util.module_from_spec(spec)
        sys.modules['ptune'] = ptune_module

        # provide DUMMY locally just for the execution
        sys.modules['petals.utils'] = MockPackage()
        sys.modules['petals.utils.misc'] = MockPackage()
        DUMMY = torch.empty(0)
        sys.modules['petals.utils.misc'].DUMMY = DUMMY

        spec.loader.exec_module(ptune_module)

        PTuneMixin = ptune_module.PTuneMixin
        DUMMY_REF = ptune_module.DUMMY

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=3
        )
        model = DummyModel(config)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        assert prompts.shape == (batch_size, 5, 16)
        assert (isinstance(intermediate_prompts, torch.Tensor) and intermediate_prompts.numel() == 0) or intermediate_prompts is DUMMY_REF

def test_deep_ptune_mode():
    mock_modules = {
        'hivemind': MockPackage(),
        'petals': MockPackage(),
    }

    with mock.patch.dict('sys.modules', mock_modules):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune_module = importlib.util.module_from_spec(spec)
        sys.modules['ptune'] = ptune_module

        sys.modules['petals.utils'] = MockPackage()
        sys.modules['petals.utils.misc'] = MockPackage()
        DUMMY = torch.empty(0)
        sys.modules['petals.utils.misc'].DUMMY = DUMMY

        spec.loader.exec_module(ptune_module)

        PTuneMixin = ptune_module.PTuneMixin

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=3
        )
        model = DummyModel(config)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        assert prompts.shape == (batch_size, 5, 16)
        # Shape should be (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
        assert intermediate_prompts.shape == (3, batch_size, 5, 16)

        # Check that the first layer (dim=0, index=0) is zeros
        assert torch.all(intermediate_prompts[0] == 0)

if __name__ == "__main__":
    pytest.main([__file__])
