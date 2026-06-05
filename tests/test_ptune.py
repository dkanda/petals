import sys
import os

sys.path.insert(0, os.path.abspath('src'))

import torch
import torch.nn as nn
from unittest import mock

def test_ptune_intermediate_embeddings_shape():
    # To test ptune.py directly we can mock the parent petals module
    # or mock specific parts. The memory states:
    # "When standard imports of submodules (e.g., from petals.client.ptune import PTuneMixin) fail in unit tests due to unresolvable dependencies in parent __init__.py files, you can bypass the parent initialization code by mocking the parent packages in sys.modules with __path__ = ["src/parent"] and __spec__ = None before performing the import."

    with mock.patch.dict('sys.modules', {}):
        petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
        sys.modules['petals'] = petals_mock

        petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
        sys.modules['petals.client'] = petals_client_mock
        petals_mock.client = petals_client_mock

        # mock hivemind
        hivemind_mock = mock.MagicMock()
        hivemind_mock.PeerID = mock.MagicMock()
        hivemind_mock.MSGPackSerializer = mock.MagicMock()
        hivemind_mock.get_logger = mock.MagicMock()
        sys.modules['hivemind'] = hivemind_mock

        # mock petals.utils.misc.DUMMY
        petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
        sys.modules['petals.utils'] = petals_utils_mock

        petals_utils_misc_mock = mock.MagicMock()
        petals_utils_misc_mock.DUMMY = torch.empty(0)
        sys.modules['petals.utils.misc'] = petals_utils_misc_mock
        petals_utils_mock.misc = petals_utils_misc_mock

        from petals.client.ptune import PTuneMixin
        from transformers import PretrainedConfig

        class MockConfig(PretrainedConfig):
            def __init__(self, tuning_mode, pre_seq_len, hidden_size, num_hidden_layers, **kwargs):
                super().__init__(**kwargs)
                self.tuning_mode = tuning_mode
                self.pre_seq_len = pre_seq_len
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers

        class MockModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32
                self.init_prompts(config)

        config = MockConfig("deep_ptune", 5, 10, 4)
        model = MockModel(config)

        assert model.prompt_embeddings.weight.shape == (5, 10)
        assert model.intermediate_prompt_embeddings.weight.shape == (5, 30)

        prompts, intermediate_prompts = model.get_prompt(2)
        assert prompts.shape == (2, 5, 10)
        assert intermediate_prompts.shape == (3, 2, 5, 10)
