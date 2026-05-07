import sys
import types
from unittest import mock
import importlib.util

def test_ptune_deep_ptune_shape():
    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    # Mock petals to bypass deep dependency import errors
    mock_petals = types.ModuleType("petals")
    mock_petals.__path__ = []
    mock_petals.utils = types.ModuleType("petals.utils")
    mock_petals.utils.misc = types.ModuleType("petals.utils.misc")
    mock_petals.utils.misc.DUMMY = None

    # Mock hivemind to bypass missing deep dependencies
    mock_hivemind = types.ModuleType("hivemind")
    mock_hivemind.__path__ = []
    mock_hivemind.get_logger = mock.MagicMock()

    with mock.patch.dict("sys.modules", {
        "petals": mock_petals,
        "petals.utils": mock_petals.utils,
        "petals.utils.misc": mock_petals.utils.misc,
        "hivemind": mock_hivemind
    }):
        spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune)

        class MockPTuneModel(ptune.PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=4
        )

        with mock.patch.object(ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
            model = MockPTuneModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)
            assert prompts.shape == (2, 5, 16)
            assert intermediate_prompts.shape == (3, 2, 5, 16)
