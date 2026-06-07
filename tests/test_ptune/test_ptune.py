import sys
import os
import pytest
from unittest import mock
import torch

sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

@pytest.fixture(autouse=True)
def mock_dependencies():
    hivemind_mock = mock.MagicMock(__path__=["mock_hivemind"], __spec__=None)
    hivemind_mock.PeerID = mock.MagicMock()
    hivemind_mock.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.get_logger = mock.MagicMock()

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.moe': mock.MagicMock(__path__=["mock_hivemind_moe"], __spec__=None),
        'hivemind.moe.client': mock.MagicMock(__path__=["mock"], __spec__=None),
        'hivemind.moe.client.remote_expert_worker': mock.MagicMock(__path__=["mock"], __spec__=None),
        'hivemind.moe.expert_uid': mock.MagicMock(__path__=["mock"], __spec__=None),
        'hivemind.p2p': mock.MagicMock(__path__=["mock"], __spec__=None),
        'hivemind.p2p.p2p_daemon': mock.MagicMock(__path__=["mock"], __spec__=None),
        'hivemind.p2p.p2p_daemon_bindings': mock.MagicMock(__path__=["mock"], __spec__=None),
        'hivemind.p2p.p2p_daemon_bindings.datastructures': mock.MagicMock(__path__=["mock"], __spec__=None),
        'hivemind.utils': mock.MagicMock(__path__=["mock"], __spec__=None),
        'hivemind.dht': mock.MagicMock(__path__=["mock"], __spec__=None),
        'hivemind.proto': mock.MagicMock(__path__=["mock"], __spec__=None),
        'hivemind.compression': mock.MagicMock(__path__=["mock"], __spec__=None),
        'hivemind.utils.tensor_descr': mock.MagicMock(__path__=["mock"], __spec__=None),
        'hivemind.utils.streaming': mock.MagicMock(__path__=["mock"], __spec__=None),
        'hivemind.utils.logging': mock.MagicMock(__path__=["mock"], __spec__=None),
        'tensor_parallel': mock.MagicMock(),
    }

    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    mocks["petals.client.inference_session"] = petals_client_mock
    mocks["petals.client.remote_sequential"] = petals_client_mock
    mocks["petals.client.routing"] = petals_client_mock

    with mock.patch.dict('sys.modules', mocks):
        with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
            yield

def test_ptune_intermediate_prompts_shape():
    from petals.client.ptune import PTuneMixin, PTuneConfig
    from transformers import PretrainedConfig
    from petals.utils.misc import DUMMY

    class DummyConfig(PretrainedConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = 64
            self.num_hidden_layers = 12
            self.tuning_mode = "deep_ptune"
            self.pre_seq_len = 5

    class DummyModel(PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = mock.MagicMock()
            self.word_embeddings.weight = mock.MagicMock()
            self.word_embeddings.weight.device = torch.device("cpu")
            self.word_embeddings.weight.dtype = torch.float32
            self.init_prompts(config)

    config = DummyConfig()
    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == torch.Size([batch_size, config.pre_seq_len, config.hidden_size])
    assert intermediate_prompts.shape == torch.Size([config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size])
