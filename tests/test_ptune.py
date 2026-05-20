import os
import sys
sys.path.insert(0, os.path.abspath('src'))
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from unittest import mock

import hivemind
hivemind.PeerID = hivemind.p2p.PeerID
hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
hivemind.get_logger = hivemind.utils.get_logger

def test_ptune_intermediate_prompts_shape():
    with mock.patch.dict('sys.modules', {}):
        sys.modules['hivemind'] = hivemind

        petals_client_inference_session_mock = mock.MagicMock()
        sys.modules['petals.client.inference_session'] = petals_client_inference_session_mock

        petals_client_remote_sequential_mock = mock.MagicMock()
        sys.modules['petals.client.remote_sequential'] = petals_client_remote_sequential_mock

        petals_client_routing_mock = mock.MagicMock()
        sys.modules['petals.client.routing'] = petals_client_routing_mock

        from petals.client.ptune import PTuneMixin

        class MockConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.hidden_size = 16
                self.num_hidden_layers = 4
                self.pre_seq_len = 5
                self.tuning_mode = "deep_ptune"

        class MockModel(PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = MockConfig()
        model = MockModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 16])
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16])
