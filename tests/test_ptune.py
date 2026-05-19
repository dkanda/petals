import os
import sys
from unittest import mock

def test_deep_ptune_intermediate_prompts_shape():
    sys.path.insert(0, os.path.abspath('src'))

    # Mock petals to avoid circular/heavy imports
    petals_mock = mock.MagicMock()
    petals_mock.__path__ = ["src/petals"]
    petals_mock.__spec__ = None
    sys.modules['petals'] = petals_mock

    petals_client_mock = mock.MagicMock()
    petals_client_mock.__path__ = ["src/petals/client"]
    petals_client_mock.__spec__ = None
    sys.modules['petals.client'] = petals_client_mock
    petals_mock.client = petals_client_mock

    # Mock utils since ptune imports from there
    petals_utils_mock = mock.MagicMock()
    petals_utils_mock.__path__ = ["src/petals/utils"]
    petals_utils_mock.__spec__ = None
    sys.modules['petals.utils'] = petals_utils_mock

    import torch
    import torch.nn as nn
    petals_utils_misc_mock = mock.MagicMock()
    petals_utils_misc_mock.DUMMY = torch.empty(0)
    sys.modules['petals.utils.misc'] = petals_utils_misc_mock

    # Polyfill hivemind things ptune needs directly
    hivemind_mock = mock.MagicMock()
    hivemind_mock.__path__ = []
    hivemind_mock.__spec__ = None
    hivemind_mock.get_logger = mock.MagicMock()
    sys.modules['hivemind'] = hivemind_mock

    import petals.client.ptune as ptune_module
    PTuneMixin = ptune_module.PTuneMixin

    from transformers import PretrainedConfig

    class MockModel(PTuneMixin, nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(10, config.hidden_size)
            self.init_prompts(config)

    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=8
    )

    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 8, 64)
    assert intermediate_prompts.shape == (3, 2, 8, 64) # num_hidden_layers - 1
