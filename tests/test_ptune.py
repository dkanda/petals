import sys
import os
sys.path.insert(0, os.path.abspath('src'))
from unittest import mock

def test_ptune_shapes():
    hivemind_mock = mock.MagicMock()

    mocks = {
        'hivemind': hivemind_mock,
        'petals.client.inference_session': mock.MagicMock(),
        'petals.client.remote_sequential': mock.MagicMock(),
        'petals.client.routing': mock.MagicMock()
    }

    with mock.patch.dict('sys.modules', mocks):
        sys.modules['petals'] = mock.MagicMock(__path__=["src/petals"], __spec__=None)
        sys.modules['petals.client'] = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
        sys.modules['petals.utils'] = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
        sys.modules['petals.utils.misc'] = mock.MagicMock()
        import torch
        sys.modules['petals.utils.misc'].DUMMY = torch.empty(0)

        import torch.nn as nn
        from transformers import PretrainedConfig
        from petals.client.ptune import PTuneConfig, PTuneMixin

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=4,
            hidden_size=8,
            num_hidden_layers=3
        )

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 4, 8])
        assert intermediate_prompts.shape == torch.Size([2, 2, 4, 8])
