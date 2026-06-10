import sys
import os
sys.path.insert(0, os.path.abspath('src'))

from unittest import mock
import torch
import torch.nn as nn

def test_ptune_intermediate_prompts_shape():
    hivemind_mock = mock.MagicMock()
    hivemind_mock.dht = mock.MagicMock()
    hivemind_mock.p2p = mock.MagicMock()
    hivemind_mock.utils = mock.MagicMock()
    hivemind_mock.utils.logging = mock.MagicMock()
    hivemind_mock.moe = mock.MagicMock()
    hivemind_mock.proto = mock.MagicMock()

    hivemind_mock.PeerID = mock.MagicMock()
    hivemind_mock.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.get_logger = mock.MagicMock()

    tensor_parallel_mock = mock.MagicMock()

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_mock.client = petals_client_mock
    petals_client_mock.inference_session = mock.MagicMock()
    petals_client_mock.remote_sequential = mock.MagicMock()
    petals_client_mock.routing = mock.MagicMock()

    petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_mock.utils = petals_utils_mock
    petals_utils_mock.misc = mock.MagicMock()
    petals_utils_mock.misc.DUMMY = torch.empty(0)

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.dht': hivemind_mock.dht,
        'hivemind.p2p': hivemind_mock.p2p,
        'hivemind.utils': hivemind_mock.utils,
        'hivemind.utils.logging': hivemind_mock.utils.logging,
        'hivemind.moe': hivemind_mock.moe,
        'hivemind.proto': hivemind_mock.proto,
        'tensor_parallel': tensor_parallel_mock,
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.client.inference_session': petals_client_mock.inference_session,
        'petals.client.remote_sequential': petals_client_mock.remote_sequential,
        'petals.client.routing': petals_client_mock.routing,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_mock.misc,
    }

    with mock.patch.dict('sys.modules', mocks):
        import petals.client.ptune as ptune

        class MockConfig:
            def __init__(self, tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=4):
                self.tuning_mode = tuning_mode
                self.pre_seq_len = pre_seq_len
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers

        class MockModel(ptune.PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        config = MockConfig()
        model = MockModel(config)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        # Check prompts shape
        assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

        # Check intermediate_prompts shape
        # In deep_ptune, intermediate_prompts is sized for num_hidden_layers - 1
        expected_intermediate_shape = (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
        assert intermediate_prompts.shape == expected_intermediate_shape, \
            f"Expected {expected_intermediate_shape}, got {intermediate_prompts.shape}"

def test_ptune_not_deep_ptune():
    hivemind_mock = mock.MagicMock()
    hivemind_mock.dht = mock.MagicMock()
    hivemind_mock.p2p = mock.MagicMock()
    hivemind_mock.utils = mock.MagicMock()
    hivemind_mock.utils.logging = mock.MagicMock()
    hivemind_mock.moe = mock.MagicMock()
    hivemind_mock.proto = mock.MagicMock()

    hivemind_mock.PeerID = mock.MagicMock()
    hivemind_mock.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.get_logger = mock.MagicMock()

    tensor_parallel_mock = mock.MagicMock()

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_mock.client = petals_client_mock
    petals_client_mock.inference_session = mock.MagicMock()
    petals_client_mock.remote_sequential = mock.MagicMock()
    petals_client_mock.routing = mock.MagicMock()

    petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_mock.utils = petals_utils_mock
    petals_utils_mock.misc = mock.MagicMock()
    petals_utils_mock.misc.DUMMY = torch.empty(0)

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.dht': hivemind_mock.dht,
        'hivemind.p2p': hivemind_mock.p2p,
        'hivemind.utils': hivemind_mock.utils,
        'hivemind.utils.logging': hivemind_mock.utils.logging,
        'hivemind.moe': hivemind_mock.moe,
        'hivemind.proto': hivemind_mock.proto,
        'tensor_parallel': tensor_parallel_mock,
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.client.inference_session': petals_client_mock.inference_session,
        'petals.client.remote_sequential': petals_client_mock.remote_sequential,
        'petals.client.routing': petals_client_mock.routing,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_mock.misc,
    }

    with mock.patch.dict('sys.modules', mocks):
        import petals.client.ptune as ptune

        class MockConfig:
            def __init__(self, tuning_mode="ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=4):
                self.tuning_mode = tuning_mode
                self.pre_seq_len = pre_seq_len
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers

        class MockModel(ptune.PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        config = MockConfig()
        model = MockModel(config)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        # Check prompts shape
        assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

        # Check intermediate_prompts is DUMMY
        assert id(intermediate_prompts) == id(petals_utils_mock.misc.DUMMY)
