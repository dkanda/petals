import torch
import torch.nn as nn
from transformers import PretrainedConfig
import sys
import os
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_intermediate_prompts_shape():
    hivemind_mock = mock.MagicMock()
    hivemind_mock.p2p.PeerID = mock.MagicMock()
    hivemind_mock.utils.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.utils.get_logger = mock.MagicMock()

    mocks = {
        'hivemind': hivemind_mock,
        'petals.client.inference_session': mock.MagicMock(),
        'petals.client.remote_sequential': mock.MagicMock(),
        'petals.client.routing': mock.MagicMock(),
        'petals.client.config': mock.MagicMock(),
        'petals': mock.MagicMock(__path__=["src/petals"], __spec__=None),
        'petals.client': mock.MagicMock(__path__=["src/petals/client"], __spec__=None),
        'petals.utils': mock.MagicMock(__path__=["src/petals/utils"], __spec__=None),
    }

    with mock.patch.dict('sys.modules', mocks):
        from petals.utils.misc import DUMMY
        from petals.client.ptune import PTuneMixin, PTuneConfig

        class DummyConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.tuning_mode = kwargs.get('tuning_mode', None)
                self.pre_seq_len = kwargs.get('pre_seq_len', 0)
                self.hidden_size = kwargs.get('hidden_size', 16)
                self.num_hidden_layers = kwargs.get('num_hidden_layers', 4)

        class ModelMock(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        config = DummyConfig(tuning_mode='deep_ptune', pre_seq_len=5, hidden_size=16, num_hidden_layers=4)
        model = ModelMock(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        # Test exact expected shapes: (batch_size, pre_seq_len, hidden_size) for prompts
        assert prompts.shape == torch.Size([2, 5, 16])
        # Test exact expected shapes: (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size) for intermediate_prompts
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16])
