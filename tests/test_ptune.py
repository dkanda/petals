import sys
import os
sys.path.insert(0, os.path.abspath('src'))
from unittest import mock
import torch
import torch.nn as nn
import hivemind

hivemind.PeerID = hivemind.p2p.PeerID
hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
hivemind.get_logger = hivemind.utils.get_logger

def test_ptune_shapes():
    # Mock heavy dependencies
    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_mock.client = petals_client_mock

    misc_mock = mock.MagicMock(__path__=["src/petals/utils/misc"], __spec__=None)
    misc_mock.DUMMY = torch.empty(0)

    # Mock sys.modules to prevent deep dependencies from loading
    mocks = {
        'petals.client.inference_session': mock.MagicMock(__path__=["src/petals/client/inference_session"], __spec__=None),
        'petals.client.remote_sequential': mock.MagicMock(__path__=["src/petals/client/remote_sequential"], __spec__=None),
        'petals.client.routing': mock.MagicMock(__path__=["src/petals/client/routing"], __spec__=None),
        'petals.utils.misc': misc_mock
    }

    with mock.patch.dict('sys.modules', mocks):
        import petals.client.ptune as ptune
        with mock.patch.object(ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
            class DummyModel(ptune.PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = nn.Embedding(100, config.hidden_size)
                    self.init_prompts(config)

            config = mock.MagicMock()
            config.tuning_mode = "deep_ptune"
            config.pre_seq_len = 5
            config.hidden_size = 16
            config.num_hidden_layers = 12

            model = DummyModel(config)

            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert prompts.shape == (2, 5, 16)
            assert intermediate_prompts.shape == (11, 2, 5, 16)
