import os
import sys
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

class MockPTuneConfig(PretrainedConfig):
    def __init__(self, tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=10, num_hidden_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(100, 10)

def test_ptune_layer_shapes():
    hivemind_mock = mock.MagicMock(__path__=["hivemind"], __spec__=None)
    hivemind_mock.utils = mock.MagicMock(__path__=["hivemind.utils"], __spec__=None)
    hivemind_mock.p2p = mock.MagicMock(__path__=["hivemind.p2p"], __spec__=None)
    hivemind_mock.dht = mock.MagicMock(__path__=["hivemind.dht"], __spec__=None)
    hivemind_mock.moe = mock.MagicMock(__path__=["hivemind.moe"], __spec__=None)
    hivemind_mock.moe.client = mock.MagicMock(__path__=["hivemind.moe.client"], __spec__=None)
    hivemind_mock.utils.get_logger = mock.MagicMock()
    hivemind_mock.utils.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.p2p.PeerID = mock.MagicMock()
    hivemind_mock.PeerID = hivemind_mock.p2p.PeerID
    hivemind_mock.MSGPackSerializer = hivemind_mock.utils.MSGPackSerializer
    hivemind_mock.get_logger = hivemind_mock.utils.get_logger

    with mock.patch.dict('sys.modules', {
        'hivemind': hivemind_mock,
        'hivemind.utils': hivemind_mock.utils,
        'hivemind.utils.logging': mock.MagicMock(__path__=[], __spec__=None),
        'hivemind.p2p': hivemind_mock.p2p,
        'hivemind.dht': hivemind_mock.dht,
        'hivemind.moe': hivemind_mock.moe,
        'hivemind.moe.client': hivemind_mock.moe.client,
        'hivemind.moe.client.remote_expert_worker': mock.MagicMock(__path__=[], __spec__=None),
        'hivemind.moe.expert_uid': mock.MagicMock(__path__=[], __spec__=None),
        'hivemind.compression': mock.MagicMock(__path__=[], __spec__=None),
        'hivemind.compression.quantization': mock.MagicMock(__path__=[], __spec__=None),
        'petals.client.inference_session': mock.MagicMock(__path__=[], __spec__=None),
        'petals.client.remote_sequential': mock.MagicMock(__path__=[], __spec__=None),
        'petals.client.routing': mock.MagicMock(__path__=[], __spec__=None),
    }), mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
        from petals.client.ptune import PTuneMixin
        import petals.client.ptune as ptune

        class ModelWithPTune(MockModel, PTuneMixin):
            pass

        model = ModelWithPTune()
        config = MockPTuneConfig()
        model.config = config

        with mock.patch.object(ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
            model.init_prompts(config)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        assert prompts.shape == (2, 5, 10), "Prompts should be (batch_size, pre_seq_len, hidden_size)"
        assert intermediate_prompts.shape == (3, 2, 5, 10), "Intermediate prompts should be (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)"
