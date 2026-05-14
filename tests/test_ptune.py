import sys
import types
from unittest import mock
import torch
import torch.nn as nn

def test_ptune_shapes():
    hivemind_mock = types.ModuleType("hivemind")
    hivemind_mock.__path__ = []
    hivemind_mock.__spec__ = None
    hivemind_mock.get_logger = mock.MagicMock()

    petals_mock = types.ModuleType("petals")
    petals_mock.__path__ = ["src/petals"]
    petals_mock.__spec__ = None

    petals_client_mock = types.ModuleType("petals.client")
    petals_client_mock.__path__ = ["src/petals/client"]
    petals_client_mock.__spec__ = None

    petals_utils_mock = types.ModuleType("petals.utils")
    petals_utils_mock.__path__ = ["src/petals/utils"]
    petals_utils_mock.__spec__ = None

    petals_utils_misc_mock = types.ModuleType("petals.utils.misc")
    petals_utils_misc_mock.DUMMY = torch.empty(0)

    with mock.patch.dict('sys.modules', {
        'hivemind': hivemind_mock,
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
    }):
        from petals.client.ptune import PTuneMixin

        class Config:
            tuning_mode = "deep_ptune"
            pre_seq_len = 5
            hidden_size = 16
            num_hidden_layers = 3

        class DummyModel(PTuneMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Config()
                self.word_embeddings = nn.Embedding(100, self.config.hidden_size)
                self.init_prompts(self.config)

        model = DummyModel()
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 5, 16)
        assert intermediate_prompts.shape == (3, 2, 5, 16)
