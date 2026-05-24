import sys
import os
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

import hivemind
hivemind.PeerID = hivemind.p2p.PeerID
hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
hivemind.get_logger = hivemind.utils.get_logger

def test_ptune_intermediate_prompt_shape():
    with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
        with mock.patch.dict('sys.modules', {
            'petals.client.inference_session': mock.MagicMock(),
            'petals.client.remote_sequential': mock.MagicMock(),
            'petals.client.routing': mock.MagicMock()
        }):
            from petals.client.ptune import PTuneMixin

            class DummyModel(PTuneMixin, nn.Module):
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
