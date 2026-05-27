import sys
import os
import torch
import torch.nn as nn
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_mixin_deep_ptune():
    hivemind_mock = mock.MagicMock(__path__=[], __spec__=None)
    hivemind_mock.p2p.PeerID = mock.MagicMock()
    hivemind_mock.PeerID = hivemind_mock.p2p.PeerID
    hivemind_mock.utils.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.MSGPackSerializer = hivemind_mock.utils.MSGPackSerializer
    hivemind_mock.utils.get_logger = mock.MagicMock()
    hivemind_mock.get_logger = hivemind_mock.utils.get_logger

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_utils_misc_mock = mock.MagicMock()

    petals_utils_misc_mock.DUMMY = torch.empty(0)

    petals_mock.client = petals_client_mock
    petals_mock.utils = petals_utils_mock
    petals_utils_mock.misc = petals_utils_misc_mock

    with mock.patch.dict('sys.modules', {
        'hivemind': hivemind_mock,
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
    }):
        import petals.client.ptune as ptune

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        class DummyConfig:
            hidden_size = 64
            num_hidden_layers = 12
            tuning_mode = "deep_ptune"
            pre_seq_len = 5

        config = DummyConfig()
        model = DummyModel(config)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert intermediate_prompts.shape == (11, 2, 5, 64)
