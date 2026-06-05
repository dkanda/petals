import sys
import os
sys.path.insert(0, os.path.abspath('src'))

from unittest import mock
import torch

def test_ptune_mixin_intermediate_prompts_shape():
    hivemind_mock = mock.MagicMock()
    hivemind_mock.p2p = mock.MagicMock()
    hivemind_mock.p2p.PeerID = mock.MagicMock()
    hivemind_mock.utils = mock.MagicMock()
    hivemind_mock.utils.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.utils.get_logger = mock.MagicMock()
    hivemind_mock.PeerID = hivemind_mock.p2p.PeerID
    hivemind_mock.MSGPackSerializer = hivemind_mock.utils.MSGPackSerializer
    hivemind_mock.get_logger = hivemind_mock.utils.get_logger

    tensor_parallel_mock = mock.MagicMock()

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.p2p': hivemind_mock.p2p,
        'hivemind.utils': hivemind_mock.utils,
        'tensor_parallel': tensor_parallel_mock,
        'petals.client.inference_session': mock.MagicMock(),
        'petals.client.remote_sequential': mock.MagicMock(),
        'petals.client.routing': mock.MagicMock(),
        'petals': mock.MagicMock(__path__=["src/petals"], __spec__=None),
        'petals.client': mock.MagicMock(__path__=["src/petals/client"], __spec__=None),
        'petals.utils': mock.MagicMock(__path__=["src/petals/utils"], __spec__=None),
        'petals.utils.misc': mock.MagicMock(__path__=["src/petals/utils/misc"], __spec__=None),
    }

    mocks['petals.utils.misc'].DUMMY = torch.empty(0)

    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin
        from transformers import PretrainedConfig
        import torch.nn as nn

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            num_hidden_layers=4,
            hidden_size=16
        )

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 16])
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16])
