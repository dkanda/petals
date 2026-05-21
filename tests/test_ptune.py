import os
import sys
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

import torch
import torch.nn as nn
from transformers import PretrainedConfig

import transformers.utils.import_utils
transformers.utils.import_utils.is_torch_fx_available = lambda: False

import hivemind
import hivemind.utils

def test_ptune_intermediate_prompts_shape():
    hivemind.PeerID = mock.MagicMock()
    hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
    hivemind.get_logger = hivemind.utils.get_logger

    mock_inference_session = mock.MagicMock(__path__=[], __spec__=None)
    mock_remote_sequential = mock.MagicMock(__path__=[], __spec__=None)
    mock_routing = mock.MagicMock(__path__=[], __spec__=None)

    with mock.patch.dict('sys.modules', {
        'petals.client.inference_session': mock_inference_session,
        'petals.client.remote_sequential': mock_remote_sequential,
        'petals.client.routing': mock_routing,
    }):
        import petals.client.ptune
        with mock.patch.object(petals.client.ptune, '_original_register_parameter', nn.Module.register_parameter):
            class DummyModel(nn.Module, petals.client.ptune.PTuneMixin):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            config = PretrainedConfig()
            config.tuning_mode = "deep_ptune"
            config.pre_seq_len = 5
            config.hidden_size = 16
            config.num_hidden_layers = 4

            model = DummyModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert prompts.shape == torch.Size([2, 5, 16])
            assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16])
