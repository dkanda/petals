import sys
import os
sys.path.insert(0, os.path.abspath('src'))
import torch
import torch.nn as nn
from unittest import mock
import hivemind
import pytest

def test_ptune_shapes():
    # Setup hivemind polyfills
    hivemind.PeerID = hivemind.p2p.PeerID
    hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
    hivemind.get_logger = hivemind.utils.get_logger

    mocks = {
        'petals.client.inference_session': mock.MagicMock(),
        'petals.client.remote_sequential': mock.MagicMock(),
        'petals.client.routing': mock.MagicMock(),
    }

    with mock.patch.dict('sys.modules', mocks):
        with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
            import petals.client.ptune as ptune

            class Config:
                tuning_mode = "deep_ptune"
                pre_seq_len = 5
                hidden_size = 10
                num_hidden_layers = 3

            class MockWordEmbeddings:
                weight = torch.zeros(1, dtype=torch.float32)

            class TestPTuneMixin(ptune.PTuneMixin):
                def __init__(self):
                    self.config = Config()
                    self.word_embeddings = MockWordEmbeddings()

            mixin = TestPTuneMixin()

            with mock.patch.object(ptune, '_original_register_parameter', nn.Module.register_parameter):
                mixin.init_prompts(mixin.config)

                batch_size = 2
                prompts, intermediate_prompts = mixin.get_prompt(batch_size)

                assert prompts.shape == torch.Size([2, 5, 10])
                assert intermediate_prompts.shape == torch.Size([2, 2, 5, 10])
