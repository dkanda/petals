import sys
import os
import pytest

sys.path.insert(0, os.path.abspath('src'))

os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

from unittest import mock
import torch
from transformers import PretrainedConfig

def test_deep_ptune_intermediate_prompts_shape():
    hivemind_mock = mock.MagicMock()
    hivemind_mock.p2p = mock.MagicMock()
    hivemind_mock.p2p.PeerID = mock.MagicMock()
    hivemind_mock.utils = mock.MagicMock()
    hivemind_mock.utils.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.utils.get_logger = mock.MagicMock()
    hivemind_mock.utils.logging = mock.MagicMock()
    hivemind_mock.PeerID = hivemind_mock.p2p.PeerID
    hivemind_mock.MSGPackSerializer = hivemind_mock.utils.MSGPackSerializer
    hivemind_mock.get_logger = hivemind_mock.utils.get_logger
    hivemind_mock.dht = mock.MagicMock()
    hivemind_mock.moe = mock.MagicMock()
    hivemind_mock.moe.expert_uid = mock.MagicMock()
    hivemind_mock.proto = mock.MagicMock()

    tensor_parallel_mock = mock.MagicMock()

    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_client_mock.inference_session = mock.MagicMock()
    petals_client_mock.remote_sequential = mock.MagicMock()
    petals_client_mock.routing = mock.MagicMock()

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_mock.client = petals_client_mock
    petals_mock.utils = mock.MagicMock()
    petals_mock.utils.misc = mock.MagicMock()
    petals_mock.utils.misc.DUMMY = torch.empty(0)

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.dht': hivemind_mock.dht,
        'hivemind.moe': hivemind_mock.moe,
        'hivemind.moe.expert_uid': hivemind_mock.moe.expert_uid,
        'hivemind.p2p': hivemind_mock.p2p,
        'hivemind.proto': hivemind_mock.proto,
        'hivemind.utils': hivemind_mock.utils,
        'hivemind.utils.logging': hivemind_mock.utils.logging,
        'tensor_parallel': tensor_parallel_mock,
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.client.inference_session': petals_client_mock.inference_session,
        'petals.client.remote_sequential': petals_client_mock.remote_sequential,
        'petals.client.routing': petals_client_mock.routing,
        'petals.utils': petals_mock.utils,
        'petals.utils.misc': petals_mock.utils.misc,
    }

    with mock.patch.dict('sys.modules', mocks):
        # We need to import the module directly to avoid petals.__init__ dependency chains
        import importlib.util
        spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

        class MockModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32
                self.init_prompts(config)

        config = PretrainedConfig()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 10
        config.hidden_size = 32
        config.num_hidden_layers = 5

        model = MockModel(config)

        # intermediate_prompt_embeddings weight shape should be (pre_seq_len, (num_hidden_layers - 1) * hidden_size)
        assert model.intermediate_prompt_embeddings.weight.shape == torch.Size([10, 128])

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

        # intermediate_prompts output shape should be (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
        assert intermediate_prompts.shape == torch.Size([4, 2, 10, 32])

def test_ptune_intermediate_prompts_shape():
    hivemind_mock = mock.MagicMock()
    hivemind_mock.p2p = mock.MagicMock()
    hivemind_mock.p2p.PeerID = mock.MagicMock()
    hivemind_mock.utils = mock.MagicMock()
    hivemind_mock.utils.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.utils.get_logger = mock.MagicMock()
    hivemind_mock.utils.logging = mock.MagicMock()
    hivemind_mock.PeerID = hivemind_mock.p2p.PeerID
    hivemind_mock.MSGPackSerializer = hivemind_mock.utils.MSGPackSerializer
    hivemind_mock.get_logger = hivemind_mock.utils.get_logger
    hivemind_mock.dht = mock.MagicMock()
    hivemind_mock.moe = mock.MagicMock()
    hivemind_mock.moe.expert_uid = mock.MagicMock()
    hivemind_mock.proto = mock.MagicMock()

    tensor_parallel_mock = mock.MagicMock()

    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_client_mock.inference_session = mock.MagicMock()
    petals_client_mock.remote_sequential = mock.MagicMock()
    petals_client_mock.routing = mock.MagicMock()

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_mock.client = petals_client_mock
    petals_mock.utils = mock.MagicMock()
    petals_mock.utils.misc = mock.MagicMock()
    petals_mock.utils.misc.DUMMY = torch.empty(0)

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.dht': hivemind_mock.dht,
        'hivemind.moe': hivemind_mock.moe,
        'hivemind.moe.expert_uid': hivemind_mock.moe.expert_uid,
        'hivemind.p2p': hivemind_mock.p2p,
        'hivemind.proto': hivemind_mock.proto,
        'hivemind.utils': hivemind_mock.utils,
        'hivemind.utils.logging': hivemind_mock.utils.logging,
        'tensor_parallel': tensor_parallel_mock,
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.client.inference_session': petals_client_mock.inference_session,
        'petals.client.remote_sequential': petals_client_mock.remote_sequential,
        'petals.client.routing': petals_client_mock.routing,
        'petals.utils': petals_mock.utils,
        'petals.utils.misc': petals_mock.utils.misc,
    }

    with mock.patch.dict('sys.modules', mocks):
        import importlib.util
        spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

        class MockModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32
                self.init_prompts(config)

        config = PretrainedConfig()
        config.tuning_mode = "ptune"
        config.pre_seq_len = 10
        config.hidden_size = 32
        config.num_hidden_layers = 5

        model = MockModel(config)

        assert not hasattr(model, 'intermediate_prompt_embeddings')

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

        assert id(intermediate_prompts) == id(ptune.DUMMY)
