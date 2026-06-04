import sys
import os
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

# Mock external dependencies to isolate testing PTuneMixin
hivemind_mock = mock.MagicMock(__path__=["mock_hivemind"], __spec__=None)
hivemind_mock.PeerID = mock.MagicMock()
hivemind_mock.MSGPackSerializer = mock.MagicMock()
hivemind_mock.get_logger = mock.MagicMock()
hivemind_mock.anext = mock.MagicMock()
hivemind_mock.deserialize_torch_tensor = mock.MagicMock()
hivemind_mock.serialize_torch_tensor = mock.MagicMock()
hivemind_mock.DHT = mock.MagicMock()
hivemind_mock.BatchTensorDescriptor = mock.MagicMock()

tensor_parallel_mock = mock.MagicMock(__path__=["mock_tensor_parallel"], __spec__=None)

sys.modules['hivemind'] = hivemind_mock
sys.modules['hivemind.utils'] = mock.MagicMock(__path__=["mock_hivemind_utils"], __spec__=None)
sys.modules['hivemind.utils.logging'] = mock.MagicMock()
sys.modules['hivemind.moe'] = mock.MagicMock(__path__=["mock_hivemind_moe"], __spec__=None)
sys.modules['hivemind.moe.client'] = mock.MagicMock(__path__=["mock_hivemind_moe_client"], __spec__=None)
sys.modules['hivemind.moe.client.remote_expert_worker'] = mock.MagicMock()
sys.modules['tensor_parallel'] = tensor_parallel_mock
sys.modules['tensor_parallel.tensor_parallel'] = tensor_parallel_mock
sys.modules['tensor_parallel.slicing_configs'] = mock.MagicMock()

import torch
import torch.nn as nn
from transformers import PretrainedConfig

import petals.client.ptune as ptune

class DummyModel(nn.Module, ptune.PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompt_embeddings_shape():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
    )
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5

    with mock.patch.object(ptune, '_original_register_parameter', nn.Module.register_parameter):
        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert intermediate_prompts.shape == (3, 2, 5, 16), f"Expected (3, 2, 5, 16), got {intermediate_prompts.shape}"
