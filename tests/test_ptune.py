import sys
import os
from unittest import mock
import torch

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_shapes():
    hivemind_mock = mock.MagicMock(__path__=["mock_hivemind"], __spec__=None)
    hivemind_mock.get_logger = mock.MagicMock()

    transformers_mock = mock.MagicMock(__path__=["mock_transformers"], __spec__=None)
    transformers_mock.PretrainedConfig = mock.MagicMock()

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_mock.utils = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_mock.utils.misc = mock.MagicMock(__path__=["src/petals/utils/misc"], __spec__=None)
    petals_mock.utils.misc.DUMMY = torch.empty(0)

    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)

    with mock.patch.dict('sys.modules', {
        'hivemind': hivemind_mock,
        'transformers': transformers_mock,
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.utils': petals_mock.utils,
        'petals.utils.misc': petals_mock.utils.misc,
    }):
        if 'petals.client.ptune' in sys.modules:
            del sys.modules['petals.client.ptune']

        import importlib
        ptune = importlib.import_module("petals.client.ptune")

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32
                self.init_prompts(config)

        class DummyConfig:
            def __init__(self):
                self.tuning_mode = "deep_ptune"
                self.pre_seq_len = 5
                self.hidden_size = 16
                self.num_hidden_layers = 4

        config = DummyConfig()
        model = DummyModel(config)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert intermediate_prompts.shape == (3, 2, 5, 16)
