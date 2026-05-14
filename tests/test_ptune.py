import sys
import types
from unittest import mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig

def test_ptune_intermediate_prompts_shape():
    mock_hivemind = types.ModuleType("hivemind")
    mock_hivemind.__path__ = []
    mock_hivemind.__spec__ = None
    mock_hivemind.get_logger = mock.MagicMock()

    mock_dht = mock.MagicMock()

    petals_mock = types.ModuleType("petals")
    petals_mock.__path__ = ["src/petals"]
    petals_mock.__spec__ = None

    petals_client_mock = types.ModuleType("petals.client")
    petals_client_mock.__path__ = ["src/petals/client"]
    petals_client_mock.__spec__ = None
    petals_mock.client = petals_client_mock

    petals_utils_mock = types.ModuleType("petals.utils")
    petals_utils_mock.__path__ = ["src/petals/utils"]
    petals_utils_mock.__spec__ = None

    petals_utils_misc_mock = types.ModuleType("petals.utils.misc")
    petals_utils_misc_mock.DUMMY = torch.empty(0, requires_grad=True)
    petals_utils_mock.misc = petals_utils_misc_mock
    petals_mock.utils = petals_utils_mock

    with mock.patch.dict('sys.modules', {
        'hivemind': mock_hivemind,
        'hivemind.dht': mock_dht,
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
    }):
        from petals.client.ptune import PTuneMixin

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 5
        config.num_hidden_layers = 3
        config.hidden_size = 8

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 5, 8)
        assert intermediate_prompts.shape == (3, 2, 5, 8)
