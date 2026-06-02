import sys
import os
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_shapes():
    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    hivemind_mock = mock.MagicMock(__path__=["mock_hivemind"], __spec__=None)
    hivemind_utils_mock = mock.MagicMock(__path__=["mock_hivemind.utils"], __spec__=None)
    hivemind_utils_logging_mock = mock.MagicMock(__spec__=None)
    hivemind_utils_logging_mock.get_logger = mock.MagicMock()
    hivemind_utils_mock.logging = hivemind_utils_logging_mock
    hivemind_mock.utils = hivemind_utils_mock
    hivemind_dht_mock = mock.MagicMock(__path__=["mock_hivemind.dht"], __spec__=None)
    hivemind_mock.dht = hivemind_dht_mock
    hivemind_p2p_mock = mock.MagicMock(__path__=["mock_hivemind.p2p"], __spec__=None)
    hivemind_mock.p2p = hivemind_p2p_mock
    hivemind_moe_mock = mock.MagicMock(__path__=["mock_hivemind.moe"], __spec__=None)
    hivemind_moe_expert_uid_mock = mock.MagicMock(__spec__=None)
    hivemind_moe_mock.expert_uid = hivemind_moe_expert_uid_mock
    hivemind_mock.moe = hivemind_moe_mock
    hivemind_mock.PeerID = mock.MagicMock()
    hivemind_mock.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.get_logger = mock.MagicMock()

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_utils_misc_mock = mock.MagicMock(__spec__=None)
    petals_utils_misc_mock.DUMMY = torch.empty(0)
    petals_utils_mock.misc = petals_utils_misc_mock
    petals_mock.utils = petals_utils_mock

    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_client_mock.inference_session = mock.MagicMock()
    petals_client_mock.remote_sequential = mock.MagicMock()
    petals_client_mock.routing = mock.MagicMock()
    petals_mock.client = petals_client_mock

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.utils': hivemind_utils_mock,
        'hivemind.utils.logging': hivemind_utils_logging_mock,
        'hivemind.dht': hivemind_dht_mock,
        'hivemind.p2p': hivemind_p2p_mock,
        'hivemind.moe': hivemind_moe_mock,
        'hivemind.moe.expert_uid': hivemind_moe_expert_uid_mock,
        'petals.client.inference_session': petals_client_mock.inference_session,
        'petals.client.remote_sequential': petals_client_mock.remote_sequential,
        'petals.client.routing': petals_client_mock.routing,
        'petals.utils.misc': petals_utils_misc_mock,
    }

    with mock.patch.dict('sys.modules', mocks):
        with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
            os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'
            import petals.client.ptune as ptune

            with mock.patch.object(ptune, '_original_register_parameter', nn.Module.register_parameter):
                class DummyConfig(PretrainedConfig):
                    def __init__(self, **kwargs):
                        super().__init__(**kwargs)
                        self.tuning_mode = "deep_ptune"
                        self.pre_seq_len = 10
                        self.hidden_size = 64
                        self.num_hidden_layers = 4

                class MockModel(nn.Module, ptune.PTuneMixin):
                    def __init__(self, config):
                        super().__init__()
                        self.config = config
                        self.word_embeddings = nn.Embedding(100, config.hidden_size)
                        self.init_prompts(config)

                config = DummyConfig()
                model = MockModel(config)

                assert model.intermediate_prompt_embeddings.weight.shape == (10, 3 * 64)

                prompts, intermediate_prompts = model.get_prompt(batch_size=2)
                assert intermediate_prompts.shape == (3, 2, 10, 64)

if __name__ == "__main__":
    test_ptune_shapes()
    print("Success")