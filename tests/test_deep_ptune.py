import sys
import types
from unittest import mock
import importlib.util

# We do this mock logic directly within the test function to isolate it and prevent polluting the test session
def test_deep_ptune_dimensions():
    # Ensure we don't accidentally import real modules if they are missing
    class MockPackage(mock.MagicMock):
        __path__ = []
        __spec__ = None

    mock_modules = {}
    mock_modules['torch'] = mock.MagicMock()
    mock_modules['torch.nn'] = mock.MagicMock()

    mock_hivemind = types.ModuleType('hivemind')
    mock_hivemind.__path__ = []
    mock_hivemind.get_logger = mock.MagicMock()
    mock_modules['hivemind'] = mock_hivemind

    mock_transformers = mock.MagicMock()
    mock_transformers.PretrainedConfig = mock.MagicMock
    mock_modules['transformers'] = mock_transformers

    mock_petals = types.ModuleType('petals')
    mock_petals.__path__ = []
    mock_petals_utils = types.ModuleType('petals.utils')
    mock_petals_utils.__path__ = []
    mock_petals_utils_misc = mock.MagicMock()
    mock_petals_utils_misc.DUMMY = mock.MagicMock()
    mock_modules['petals'] = mock_petals
    mock_modules['petals.utils'] = mock_petals_utils
    mock_modules['petals.utils.misc'] = mock_petals_utils_misc

    # We must wrap this in a patch dict to ensure it doesn't leak or fail
    with mock.patch.dict('sys.modules', mock_modules):
        # Reload the ptune module so it picks up the mocked torch/nn
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune)

        import torch
        import torch.nn as nn

        class DummyConfig:
            def __init__(self, tuning_mode, pre_seq_len, hidden_size, num_hidden_layers):
                self.tuning_mode = tuning_mode
                self.pre_seq_len = pre_seq_len
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers

        class TestPTuneMixin(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight = mock.MagicMock()

        # Test init_prompts
        config = DummyConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=10, num_hidden_layers=4)
        mixin = TestPTuneMixin(config)
        mixin.init_prompts(config)

        # Verify the embedding size is created properly: (4 - 1) * 10 = 30
        ptune.nn.Embedding.assert_any_call(
            5, 30, dtype=torch.float32
        )

        # Test get_prompt
        mixin.prefix_tokens = mock.MagicMock()
        mixin.prompt_embeddings = mock.MagicMock()
        mixin.intermediate_prompt_embeddings = mock.MagicMock()

        mock_tensor = mock.MagicMock()
        mixin.intermediate_prompt_embeddings.return_value = mock_tensor
        mock_tensor.view.return_value = mock_tensor
        mock_tensor.permute.return_value = mock_tensor

        prompts, intermediate_prompts = mixin.get_prompt(2)

        # Verify it reshaped using num_hidden_layers - 1: batch_size=2, pre_seq_len=5, num_layers=3, hidden_size=10
        mock_tensor.view.assert_called_with(2, 5, 3, 10)
