import sys
import unittest
from unittest import mock
import torch
import torch.nn as nn

def test_ptune_mixin_deep_ptune():
    hivemind_mock = mock.MagicMock()
    transformers_mock = mock.MagicMock()

    petals_mock = mock.MagicMock()
    petals_mock.__path__ = ["src/petals"]
    petals_mock.__spec__ = None

    petals_client_mock = mock.MagicMock()
    petals_client_mock.__path__ = ["src/petals/client"]
    petals_client_mock.__spec__ = None

    petals_utils_mock = mock.MagicMock()
    petals_utils_mock.__path__ = ["src/petals/utils"]
    petals_utils_mock.__spec__ = None

    petals_utils_misc_mock = mock.MagicMock()
    petals_utils_misc_mock.__path__ = ["src/petals/utils/misc"]
    petals_utils_misc_mock.__spec__ = None
    petals_utils_misc_mock.DUMMY = torch.empty(0)

    petals_mock.utils = petals_utils_mock
    petals_mock.client = petals_client_mock
    petals_utils_mock.misc = petals_utils_misc_mock

    mocks = {
        'hivemind': hivemind_mock,
        'transformers': transformers_mock,
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
    }

    with mock.patch.dict('sys.modules', mocks):
        import petals.client.ptune as ptune_module
        from petals.client.ptune import PTuneMixin

        with mock.patch.object(ptune_module, '_original_register_parameter', nn.Module.register_parameter):
            class DummyConfig:
                def __init__(self):
                    self.tuning_mode = "deep_ptune"
                    self.pre_seq_len = 5
                    self.hidden_size = 16
                    self.num_hidden_layers = 10

            class DummyWordEmbeddings(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = nn.Parameter(torch.empty(0, dtype=torch.float32))

            class DummyModel(PTuneMixin, nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = DummyWordEmbeddings()
                    self.init_prompts(config)

            config = DummyConfig()
            model = DummyModel(config)

            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            assert prompts.shape == (2, 5, 16)
            assert intermediate_prompts.shape == (9, 2, 5, 16)
