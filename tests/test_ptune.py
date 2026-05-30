import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import torch
import torch.nn as nn
from unittest import mock
import pytest

def test_ptune_mixin_num_hidden_layers_minus_1():
    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_mock.client = petals_client_mock

    petals_utils_misc_mock = mock.MagicMock()
    petals_utils_misc_mock.DUMMY = torch.empty(0)

    hivemind_mock = mock.MagicMock()
    def fake_get_logger(name):
        return mock.MagicMock()
    hivemind_mock.get_logger = fake_get_logger

    with mock.patch.dict("sys.modules", {
        "petals": petals_mock,
        "petals.client": petals_client_mock,
        "petals.utils.misc": petals_utils_misc_mock,
        "hivemind": hivemind_mock
    }):
        import petals.client.ptune as ptune

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        class Config:
            tuning_mode = "deep_ptune"
            pre_seq_len = 5
            hidden_size = 16
            num_hidden_layers = 12

        config = Config()
        model = DummyModel(config)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 5, 16)
        assert intermediate_prompts.shape == (11, 2, 5, 16)
