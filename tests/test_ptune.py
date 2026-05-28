import sys
import os
import pytest
import torch
from unittest import mock
from transformers import PretrainedConfig

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_mixin_deep_ptune():
    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_mock.client = petals_client_mock

    with mock.patch.dict('sys.modules', {
        'hivemind': mock.MagicMock(__path__=[], __spec__=None),
        'petals.utils.misc': mock.MagicMock(DUMMY=torch.empty(0)),
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.client.inference_session': mock.MagicMock(),
        'petals.client.remote_sequential': mock.MagicMock(),
        'petals.client.routing': mock.MagicMock(),
    }):
        from petals.client.ptune import PTuneMixin, PTuneConfig, _original_register_parameter
        import petals.client.ptune as ptune

        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=4,
            num_hidden_layers=5,
            hidden_size=16
        )

        with mock.patch.object(ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
            model = DummyModel(config)
            model.init_prompts(config)

            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert prompts.shape == (2, 4, 16)
            assert intermediate_prompts.shape == (4, 2, 4, 16)
