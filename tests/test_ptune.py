import sys
import os
from unittest import mock
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_intermediate_prompts_shape():
    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_utils_mock.misc = mock.MagicMock(__path__=["src/petals/utils/misc"], __spec__=None)
    petals_utils_mock.misc.DUMMY = torch.empty(0)

    hivemind_mock = mock.MagicMock(__path__=["hivemind"], __spec__=None)
    hivemind_mock.p2p.PeerID = mock.MagicMock()
    hivemind_mock.PeerID = hivemind_mock.p2p.PeerID
    hivemind_mock.utils.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.MSGPackSerializer = hivemind_mock.utils.MSGPackSerializer
    hivemind_mock.utils.logging = mock.MagicMock(__path__=["hivemind.utils.logging"], __spec__=None)
    hivemind_mock.utils.logging.get_logger = mock.MagicMock()
    hivemind_mock.utils.get_logger = hivemind_mock.utils.logging.get_logger
    hivemind_mock.get_logger = hivemind_mock.utils.get_logger

    tensor_parallel_mock = mock.MagicMock(__path__=["tensor_parallel"], __spec__=None)

    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_client_mock.inference_session = mock.MagicMock(__path__=["src/petals/client/inference_session"], __spec__=None)
    petals_client_mock.remote_sequential = mock.MagicMock(__path__=["src/petals/client/remote_sequential"], __spec__=None)
    petals_client_mock.routing = mock.MagicMock(__path__=["src/petals/client/routing"], __spec__=None)

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.p2p': hivemind_mock.p2p,
        'hivemind.utils': hivemind_mock.utils,
        'hivemind.utils.logging': hivemind_mock.utils.logging,
        'tensor_parallel': tensor_parallel_mock,
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.client.inference_session': petals_client_mock.inference_session,
        'petals.client.remote_sequential': petals_client_mock.remote_sequential,
        'petals.client.routing': petals_client_mock.routing,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_mock.misc,
    }

    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin
        from petals.utils.misc import DUMMY

        class MockConfig:
            def __init__(self):
                self.tuning_mode = "deep_ptune"
                self.pre_seq_len = 5
                self.hidden_size = 16
                self.num_hidden_layers = 4

        class MockModel(PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        config = MockConfig()
        model = MockModel(config)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 5, 16)
        assert intermediate_prompts.shape == (3, 2, 5, 16)
