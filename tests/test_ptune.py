import sys
import os
import torch
import torch.nn as nn
from unittest import mock

def test_ptune_intermediate_shape():
    sys.path.insert(0, os.path.abspath('src'))

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_mock.client = petals_client_mock
    petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_mock.utils = petals_utils_mock
    petals_utils_misc_mock = mock.MagicMock(__path__=["src/petals/utils/misc"], __spec__=None)
    petals_utils_mock.misc = petals_utils_misc_mock
    petals_utils_misc_mock.DUMMY = torch.empty(0)

    hivemind_mock = mock.MagicMock(__path__=["hivemind"], __spec__=None)
    hivemind_utils_mock = mock.MagicMock(__path__=["hivemind/utils"], __spec__=None)
    hivemind_mock.utils = hivemind_utils_mock
    hivemind_mock.get_logger = mock.MagicMock()

    mocks = {
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
        'hivemind': hivemind_mock,
        'transformers': mock.MagicMock(__path__=[], __spec__=None),
        'transformers.utils.import_utils': mock.MagicMock(__path__=[], __spec__=None),
    }

    with mock.patch.dict('sys.modules', mocks):
        # Local import after setting up mocks
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune)

        # Test deep_ptune shape
        config = mock.MagicMock()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 5
        config.hidden_size = 16
        config.num_hidden_layers = 12

        class MockModel(ptune.PTuneMixin):
            def __init__(self):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32

        model = MockModel()

        with mock.patch.object(ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
            model.init_prompts(config)

            # Check parameter sizes
            assert model.prompt_embeddings.weight.shape == (5, 16)
            assert model.intermediate_prompt_embeddings.weight.shape == (5, (12 - 1) * 16)

            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert prompts.shape == (2, 5, 16)
            assert intermediate_prompts.shape == (12 - 1, 2, 5, 16)
