import sys
from unittest import mock
import torch
import torch.nn as nn

def test_ptune_intermediate_prompt_shape():
    # Mocking necessary modules
    petals_mock = mock.MagicMock()
    petals_mock.__path__ = ["src/petals"]
    petals_mock.__spec__ = None

    petals_client_mock = mock.MagicMock()
    petals_client_mock.__path__ = ["src/petals/client"]
    petals_client_mock.__spec__ = None
    petals_mock.client = petals_client_mock

    petals_utils_mock = mock.MagicMock()
    petals_utils_mock.__path__ = ["src/petals/utils"]
    petals_utils_mock.__spec__ = None
    petals_mock.utils = petals_utils_mock

    petals_utils_misc_mock = mock.MagicMock()
    petals_utils_misc_mock.DUMMY = torch.empty(0)
    petals_utils_mock.misc = petals_utils_misc_mock

    mocks = {
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
        'hivemind': mock.MagicMock(),
        'transformers': mock.MagicMock(),
    }

    with mock.patch.dict('sys.modules', mocks):
        import petals.client.ptune as ptune_module
        from petals.client.ptune import PTuneMixin

        class MockConfig:
            def __init__(self):
                self.pre_seq_len = 5
                self.tuning_mode = "deep_ptune"
                self.hidden_size = 8
                self.num_hidden_layers = 4

        class MockModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.tuning_mode = config.tuning_mode
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32
                self.init_prompts(config)

        # Patch _original_register_parameter because torch.nn.Module.register_parameter gets copied
        with mock.patch.object(ptune_module, '_original_register_parameter', nn.Module.register_parameter):
            config = MockConfig()
            model = MockModel(config)

            prompts, intermediate_prompts = model.get_prompt(batch_size=2)
            assert intermediate_prompts.shape == (3, 2, 5, 8)
