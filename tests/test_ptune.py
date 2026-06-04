import sys
import os
sys.path.insert(0, os.path.abspath('src'))

from unittest import mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig

def test_ptune_mixin_shapes():
    hivemind_mock = mock.MagicMock(__path__=["src/hivemind"], __spec__=None)
    with mock.patch.dict('sys.modules', {
        'hivemind': hivemind_mock,
        'tensor_parallel': mock.MagicMock(),
        'petals.utils.misc': mock.MagicMock(DUMMY=torch.empty(0)),
        'petals': mock.MagicMock(__path__=["src/petals"], __spec__=None),
        'petals.client': mock.MagicMock(__path__=["src/petals/client"], __spec__=None),
    }):
        from petals.client.ptune import PTuneMixin

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=4,
        )

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 16])
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16])
