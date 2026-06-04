import sys
import os
import torch
import torch.nn as nn
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

def test_ptune_mixin():
    hivemind_mock = mock.MagicMock(__path__=["mock_hivemind"], __spec__=None)
    hivemind_mock.PeerID = mock.MagicMock()
    hivemind_mock.utils = mock.MagicMock(__path__=["mock_hivemind_utils"], __spec__=None)
    hivemind_mock.utils.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.utils.logging = mock.MagicMock(__path__=["mock_hivemind_utils_logging"], __spec__=None)
    hivemind_mock.utils.logging.get_logger = mock.MagicMock()
    hivemind_mock.utils.get_logger = hivemind_mock.utils.logging.get_logger
    hivemind_mock.get_logger = hivemind_mock.utils.logging.get_logger
    hivemind_mock.MSGPackSerializer = hivemind_mock.utils.MSGPackSerializer
    hivemind_mock.dht = mock.MagicMock(__path__=["mock_hivemind_dht"], __spec__=None)
    hivemind_mock.dht.DHT = mock.MagicMock()
    hivemind_mock.dht.DHTNode = mock.MagicMock()
    hivemind_mock.dht.DHTValue = mock.MagicMock()
    hivemind_mock.moe = mock.MagicMock(__path__=["mock_hivemind_moe"], __spec__=None)
    hivemind_mock.moe.client = mock.MagicMock(__path__=["mock_hivemind_moe_client"], __spec__=None)
    hivemind_mock.moe.client.remote_expert_worker = mock.MagicMock(__path__=["mock_hivemind_moe_client_remote_expert_worker"], __spec__=None)
    hivemind_mock.moe.client.remote_expert_worker.RemoteExpertWorker = mock.MagicMock()
    hivemind_mock.moe.expert_uid = mock.MagicMock(__path__=["mock_hivemind_moe_expert_uid"], __spec__=None)
    hivemind_mock.moe.expert_uid.ExpertUID = mock.MagicMock()
    hivemind_mock.p2p = mock.MagicMock(__path__=["mock_hivemind_p2p"], __spec__=None)
    hivemind_mock.p2p.StubBase = mock.MagicMock()
    hivemind_mock.proto = mock.MagicMock(__path__=["mock_hivemind_proto"], __spec__=None)
    hivemind_mock.compression = mock.MagicMock(__path__=["mock_hivemind_compression"], __spec__=None)

    tensor_parallel_mock = mock.MagicMock(__path__=["mock_tensor_parallel"], __spec__=None)

    petals_client_inference_session_mock = mock.MagicMock()
    petals_client_remote_sequential_mock = mock.MagicMock()
    petals_client_routing_mock = mock.MagicMock()

    petals_utils_misc_mock = mock.MagicMock(__path__=["mock_petals_utils_misc"], __spec__=None)
    DUMMY = torch.empty(0)
    petals_utils_misc_mock.DUMMY = DUMMY
    petals_utils_misc_mock.is_dummy = lambda x: x is DUMMY

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.utils': hivemind_mock.utils,
        'hivemind.utils.logging': hivemind_mock.utils.logging,
        'hivemind.dht': hivemind_mock.dht,
        'hivemind.moe': hivemind_mock.moe,
        'hivemind.moe.client': hivemind_mock.moe.client,
        'hivemind.moe.client.remote_expert_worker': hivemind_mock.moe.client.remote_expert_worker,
        'hivemind.moe.expert_uid': hivemind_mock.moe.expert_uid,
        'hivemind.p2p': hivemind_mock.p2p,
        'hivemind.proto': hivemind_mock.proto,
        'hivemind.compression': hivemind_mock.compression,
        'tensor_parallel': tensor_parallel_mock,
        'petals.client.inference_session': petals_client_inference_session_mock,
        'petals.client.remote_sequential': petals_client_remote_sequential_mock,
        'petals.client.routing': petals_client_routing_mock,
        'petals.utils.misc': petals_utils_misc_mock,
    }

    with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
        with mock.patch.dict('sys.modules', mocks):
            from petals.client.ptune import PTuneMixin
            from transformers import PretrainedConfig

            class DummyModel(nn.Module, PTuneMixin):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            # test deep_ptune
            config_deep = PretrainedConfig(
                tuning_mode="deep_ptune",
                pre_seq_len=4,
                hidden_size=8,
                num_hidden_layers=3
            )
            model_deep = DummyModel(config_deep)
            batch_size = 2
            prompts, intermediate_prompts = model_deep.get_prompt(batch_size)

            assert prompts.shape == (2, 4, 8)
            assert intermediate_prompts.shape == (2, 2, 4, 8)

            # test ptune
            config_ptune = PretrainedConfig(
                tuning_mode="ptune",
                pre_seq_len=4,
                hidden_size=8,
                num_hidden_layers=3
            )
            model_ptune = DummyModel(config_ptune)
            prompts_ptune, intermediate_prompts_ptune = model_ptune.get_prompt(batch_size)
            assert prompts_ptune.shape == (2, 4, 8)
            assert intermediate_prompts_ptune is DUMMY
