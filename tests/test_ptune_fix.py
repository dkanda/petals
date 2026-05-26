import sys
import os
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

import torch
import torch.nn as nn

def test_deep_ptune_shapes():
    import hivemind
    hivemind.PeerID = hivemind.p2p.PeerID
    hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
    hivemind.get_logger = hivemind.utils.get_logger

    mock_petals_client_inference_session = mock.MagicMock()
    mock_petals_client_remote_sequential = mock.MagicMock()
    mock_petals_client_routing = mock.MagicMock()

    with mock.patch.dict('sys.modules', {
        'petals.client.inference_session': mock_petals_client_inference_session,
        'petals.client.remote_sequential': mock_petals_client_remote_sequential,
        'petals.client.routing': mock_petals_client_routing,
    }):
        with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
            from petals.client.ptune import PTuneMixin

            class DummyConfig:
                def __init__(self):
                    self.tuning_mode = "deep_ptune"
                    self.pre_seq_len = 5
                    self.hidden_size = 8
                    self.num_hidden_layers = 4

            class DummyModel(PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            config = DummyConfig()
            model = DummyModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert prompts.shape == (2, 5, 8)
            assert intermediate_prompts.shape == (3, 2, 5, 8)
