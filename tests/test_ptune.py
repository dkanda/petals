import sys
import os
import unittest
from unittest import mock
import torch

sys.path.insert(0, os.path.abspath('src'))

def test_ptune():
    hivemind_mock = mock.MagicMock(__path__=[], __spec__=None)
    hivemind_mock.PeerID = mock.MagicMock()
    hivemind_mock.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.get_logger = mock.MagicMock()

    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_client_mock.inference_session = mock.MagicMock(__path__=[], __spec__=None)
    petals_client_mock.remote_sequential = mock.MagicMock(__path__=[], __spec__=None)
    petals_client_mock.routing = mock.MagicMock(__path__=[], __spec__=None)

    tensor_parallel_mock = mock.MagicMock(__path__=[], __spec__=None)

    petals_utils_misc_mock = mock.MagicMock(__path__=[], __spec__=None)
    petals_utils_misc_mock.DUMMY = torch.empty(0)

    mocks = {
        'hivemind': hivemind_mock,
        'tensor_parallel': tensor_parallel_mock,
        'petals.client.inference_session': petals_client_mock.inference_session,
        'petals.client.remote_sequential': petals_client_mock.remote_sequential,
        'petals.client.routing': petals_client_mock.routing,
        'petals.utils.misc': petals_utils_misc_mock,
        'petals': mock.MagicMock(__path__=["src/petals"], __spec__=None),
        'petals.utils': mock.MagicMock(__path__=["src/petals/utils"], __spec__=None),
        'petals.client': mock.MagicMock(__path__=["src/petals/client"], __spec__=None),
    }

    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin
        from transformers import PretrainedConfig

        class MockConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.tuning_mode = kwargs.get("tuning_mode")
                self.pre_seq_len = kwargs.get("pre_seq_len", 0)
                self.hidden_size = kwargs.get("hidden_size", 64)
                self.num_hidden_layers = kwargs.get("num_hidden_layers", 10)

        class MockModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32
                with mock.patch('petals.client.ptune.force_non_empty_weights'):
                    self.init_prompts(config)

        config = MockConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=64, num_hidden_layers=10)
        model = MockModel(config)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 64])
        assert intermediate_prompts.shape == torch.Size([9, 2, 5, 64])

if __name__ == '__main__':
    test_ptune()
