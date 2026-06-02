import sys
import os
import torch
import torch.nn as nn
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

from transformers import PretrainedConfig

def test_ptune_deep_ptune():
    # As instructed by memory: polyfill missing hivemind exports
    import hivemind
    hivemind.get_logger = mock.MagicMock()

    # And polyfill other unresolvables
    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_mock.client = petals_client_mock
    petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_mock.utils = petals_utils_mock
    petals_utils_misc_mock = mock.MagicMock(__path__=["src/petals/utils/misc"], __spec__=None)
    petals_utils_misc_mock.DUMMY = torch.empty(0)
    petals_utils_mock.misc = petals_utils_misc_mock

    # We must patch these just to get past the un-buildable dependencies on the parent modules
    with mock.patch.dict('sys.modules', {
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock
    }):
        import petals.client.ptune as ptune

        class DummyConfig(PretrainedConfig):
            tuning_mode = "deep_ptune"
            pre_seq_len = 5
            hidden_size = 16
            num_hidden_layers = 10

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(100, 16)
                self.init_prompts(config)

        with mock.patch.object(ptune, '_original_register_parameter', nn.Module.register_parameter):
            config = DummyConfig()
            model = DummyModel(config)

            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            assert prompts.shape == torch.Size([2, 5, 16])
            assert intermediate_prompts.shape == torch.Size([9, 2, 5, 16])
