import sys
import types
from unittest import mock
import torch
import transformers

def test_ptune_shapes():
    # Isolate ptune.py dependencies
    hivemind_mock = mock.MagicMock()
    hivemind_mock.__path__ = []
    hivemind_mock.__spec__ = None
    get_logger_mock = mock.MagicMock()
    hivemind_mock.get_logger = get_logger_mock

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
    petals_utils_misc_mock.DUMMY = torch.tensor([0])
    petals_utils_misc_mock.__path__ = ["src/petals/utils/misc"]
    petals_utils_misc_mock.__spec__ = None

    petals_utils_mock.misc = petals_utils_misc_mock
    petals_mock.utils = petals_utils_mock
    petals_mock.client = petals_client_mock

    with mock.patch.dict('sys.modules', {
        'hivemind': hivemind_mock,
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
    }):
        from petals.client.ptune import PTuneMixin

        class DummyConfig(transformers.PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.hidden_size = 64
                self.num_hidden_layers = 4
                self.tuning_mode = "deep_ptune"
                self.pre_seq_len = 10

        mixin = PTuneMixin()
        mixin.config = DummyConfig()
        mixin.init_prompts(mixin.config)

        mixin.word_embeddings = mock.MagicMock()
        mixin.word_embeddings.weight.device = torch.device('cpu')
        mixin.word_embeddings.weight.dtype = torch.float32

        prompts, intermediate_prompts = mixin.get_prompt(batch_size=2)

        # prompts should be (batch_size, pre_seq_len, hidden_size) = (2, 10, 64)
        assert prompts.shape == torch.Size([2, 10, 64])
        # intermediate_prompts should be (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size) = (3, 2, 10, 64)
        assert intermediate_prompts.shape == torch.Size([3, 2, 10, 64])
