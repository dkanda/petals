import pytest
import sys
import importlib.util
from unittest.mock import patch, MagicMock

# This test isolates PTuneMixin logic by mocking out heavy dependencies
# to allow it to be tested in constrained CI environments.
def test_ptune():
    class MockPackage:
        pass

    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    mock_modules = {
        'petals': MockPackage(),
        'petals.utils': MockPackage(),
        'petals.utils.misc': MockPackage(),
        'hivemind': MockPackage()
    }

    mock_modules['petals.utils.misc'].DUMMY = torch.empty(0)
    mock_modules['hivemind'].get_logger = lambda name: None

    with patch.dict('sys.modules', mock_modules):
        spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules['petals.client.ptune'] = ptune
        sys.modules['petals.client'] = MockPackage()
        sys.modules['petals.client'].ptune = ptune
        spec.loader.exec_module(ptune)

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 5
        config.hidden_size = 16
        config.num_hidden_layers = 4

        with patch('petals.client.ptune._original_register_parameter', nn.Module.register_parameter):
            model = DummyModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 5, 16)
        assert intermediate_prompts.shape == (4, 2, 5, 16)
