import sys
import os
from unittest import mock

import torch

# Mock petals modules and dependencies
petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
petals_utils_misc_mock = mock.MagicMock()
petals_utils_misc_mock.DUMMY = torch.empty(0)
petals_utils_mock.misc = petals_utils_misc_mock
petals_mock.client = petals_client_mock
petals_mock.utils = petals_utils_mock

mocks = {
    'petals': petals_mock,
    'petals.client': petals_client_mock,
    'petals.utils': petals_utils_mock,
    'petals.utils.misc': petals_utils_misc_mock,
    'hivemind': mock.MagicMock(__path__=["fake_hivemind"], __spec__=None),
    'hivemind.get_logger': mock.MagicMock(),
    'transformers': mock.MagicMock(__path__=["fake_transformers"], __spec__=None),
    'transformers.PretrainedConfig': mock.MagicMock,
}

def test_ptune_mixin_deep_ptune():
    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin, force_non_empty_weights
        import torch.nn as nn

        class PretrainedConfig:
            def __init__(self, tuning_mode, pre_seq_len, hidden_size, num_hidden_layers):
                self.tuning_mode = tuning_mode
                self.pre_seq_len = pre_seq_len
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers

        class DummyWordEmbeddings:
            def __init__(self, hidden_size):
                self.weight = torch.empty((100, hidden_size))

        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = DummyWordEmbeddings(config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode='deep_ptune',
            pre_seq_len=5,
            hidden_size=8,
            num_hidden_layers=4
        )
        model = DummyModel(config)

        assert model.prompt_embeddings.weight.shape == torch.Size([5, 8])
        assert model.intermediate_prompt_embeddings.weight.shape == torch.Size([5, 24]) # 5 * (4-1)*8

        prompts, intermediate = model.get_prompt(batch_size=2)
        assert prompts.shape == torch.Size([2, 5, 8])
        assert intermediate.shape == torch.Size([3, 2, 5, 8]) # 4-1 = 3

def test_ptune_mixin_ptune():
    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin, force_non_empty_weights
        import torch.nn as nn
        from petals.utils.misc import DUMMY

        class PretrainedConfig:
            def __init__(self, tuning_mode, pre_seq_len, hidden_size, num_hidden_layers):
                self.tuning_mode = tuning_mode
                self.pre_seq_len = pre_seq_len
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers

        class DummyWordEmbeddings:
            def __init__(self, hidden_size):
                self.weight = torch.empty((100, hidden_size))

        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = DummyWordEmbeddings(config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode='ptune',
            pre_seq_len=5,
            hidden_size=8,
            num_hidden_layers=4
        )
        model = DummyModel(config)

        assert model.prompt_embeddings.weight.shape == torch.Size([5, 8])
        assert not hasattr(model, 'intermediate_prompt_embeddings')

        prompts, intermediate = model.get_prompt(batch_size=2)
        assert prompts.shape == torch.Size([2, 5, 8])
        assert intermediate is DUMMY
