import sys
import os
import torch
import torch.nn as nn
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_shapes():
    import torch
    import torch.nn as nn
    from unittest import mock

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_mock.client = petals_client_mock

    petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_utils_misc_mock = mock.MagicMock(__spec__=None)
    petals_utils_misc_mock.DUMMY = torch.empty(0)
    petals_utils_mock.misc = petals_utils_misc_mock
    petals_mock.utils = petals_utils_mock

    hivemind_mock = mock.MagicMock(__path__=[], __spec__=None)
    hivemind_utils_mock = mock.MagicMock(__spec__=None)
    hivemind_p2p_mock = mock.MagicMock(__spec__=None)

    hivemind_mock.p2p = hivemind_p2p_mock
    hivemind_mock.utils = hivemind_utils_mock

    hivemind_mock.PeerID = hivemind_p2p_mock.PeerID
    hivemind_mock.MSGPackSerializer = hivemind_utils_mock.MSGPackSerializer
    hivemind_mock.get_logger = hivemind_utils_mock.get_logger

    transformers_mock = mock.MagicMock(__path__=[], __spec__=None)
    transformers_utils_mock = mock.MagicMock(__path__=[], __spec__=None)
    transformers_utils_import_utils_mock = mock.MagicMock(__spec__=None)
    transformers_mock.utils = transformers_utils_mock
    transformers_utils_mock.import_utils = transformers_utils_import_utils_mock

    mocks = {
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.client.inference_session': mock.MagicMock(),
        'petals.client.remote_sequential': mock.MagicMock(),
        'petals.client.routing': mock.MagicMock(),
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
        'hivemind': hivemind_mock,
        'transformers': transformers_mock,
        'transformers.utils': transformers_utils_mock,
        'transformers.utils.import_utils': transformers_utils_import_utils_mock,
    }

    with mock.patch.dict('sys.modules', mocks):
        import petals.client.ptune as ptune

        class MockConfig:
            tuning_mode = "deep_ptune"
            pre_seq_len = 5
            hidden_size = 16
            num_hidden_layers = 12

        class MockModel(ptune.PTuneMixin):
            def __init__(self):
                self.config = MockConfig()
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32
                self.init_prompts(self.config)

        with mock.patch.object(ptune, '_original_register_parameter', nn.Module.register_parameter):
            model = MockModel()
            # Check correct weights shape (pre_seq_len, (num_hidden_layers - 1) * hidden_size)
            assert model.intermediate_prompt_embeddings.weight.shape == (5, 11 * 16)

            prompts, intermediate_prompts = model.get_prompt(2)

            # Check intermediate_prompts shape ((num_hidden_layers - 1), batch_size, pre_seq_len, hidden_size)
            # permuted shape will be (11, 2, 5, 16)
            assert intermediate_prompts.shape == (11, 2, 5, 16)
