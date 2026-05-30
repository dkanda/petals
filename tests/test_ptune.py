import sys
import os
from unittest import mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_intermediate_prompt_shape():
    petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_utils_misc_mock = mock.MagicMock(__path__=["src/petals/utils/misc"], __spec__=None)
    petals_utils_misc_mock.DUMMY = torch.empty(0)
    petals_utils_mock.misc = petals_utils_misc_mock

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_mock.client = petals_client_mock
    petals_mock.utils = petals_utils_mock

    mocks = {
        "hivemind": mock.MagicMock(),
        "petals": petals_mock,
        "petals.client": petals_client_mock,
        "petals.client.inference_session": mock.MagicMock(),
        "petals.client.remote_sequential": mock.MagicMock(),
        "petals.client.routing": mock.MagicMock(),
        "petals.utils": petals_utils_mock,
        "petals.utils.misc": petals_utils_misc_mock,
        "petals.utils.peft": mock.MagicMock(),
    }

    with mock.patch.dict("sys.modules", mocks):
        import petals.client.ptune as ptune

        class force_non_empty_weights:
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        class DummyModel(nn.Module, ptune.PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        with mock.patch.object(ptune, "get_logger", mock.MagicMock()), \
             mock.patch.object(ptune, "force_non_empty_weights", force_non_empty_weights):

            config = PretrainedConfig(
                hidden_size=16,
                num_hidden_layers=4,
                tuning_mode="deep_ptune",
                pre_seq_len=5
            )
            model = DummyModel(config)

            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert prompts.shape == (2, 5, 16)
            # This should fail right now!
            assert intermediate_prompts.shape == (3, 2, 5, 16)
