import sys
import os
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

def test_ptune_intermediate_prompt_shape():
    mock_hivemind = mock.MagicMock()
    mock_tensor_parallel = mock.MagicMock()

    mocks = {
        'hivemind': mock_hivemind,
        'hivemind.moe': mock_hivemind,
        'hivemind.moe.client': mock_hivemind,
        'hivemind.moe.client.remote_expert_worker': mock_hivemind,
        'hivemind.moe.server': mock_hivemind,
        'hivemind.moe.server.module_backend': mock_hivemind,
        'hivemind.moe.server.connection_handler': mock_hivemind,
        'hivemind.moe.expert_uid': mock_hivemind,
        'hivemind.p2p': mock_hivemind,
        'hivemind.p2p.p2p_daemon_bindings': mock_hivemind,
        'hivemind.p2p.p2p_daemon_bindings.control': mock_hivemind,
        'hivemind.p2p.p2p_daemon': mock_hivemind,
        'hivemind.dht': mock_hivemind,
        'hivemind.dht.node': mock_hivemind,
        'hivemind.proto': mock_hivemind,
        'hivemind.proto.runtime_pb2': mock_hivemind,
        'hivemind.utils': mock_hivemind,
        'hivemind.utils.asyncio': mock_hivemind,
        'hivemind.utils.logging': mock_hivemind,
        'hivemind.utils.mpfuture': mock_hivemind,
        'hivemind.utils.streaming': mock_hivemind,
        'hivemind.utils.nested': mock_hivemind,
        'hivemind.utils.tensor_descr': mock_hivemind,
        'hivemind.compression': mock_hivemind,
        'hivemind.compression.serialization': mock_hivemind,
        'tensor_parallel': mock_tensor_parallel,
        'tensor_parallel.slicing_configs': mock_tensor_parallel,
        'tensor_parallel.tensor_parallel': mock_tensor_parallel,
    }

    with mock.patch.dict('sys.modules', mocks):
        with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
            from petals.client.ptune import PTuneMixin

            class MockConfig(PretrainedConfig):
                def __init__(self):
                    super().__init__()
                    self.pre_seq_len = 5
                    self.hidden_size = 10
                    self.num_hidden_layers = 4
                    self.tuning_mode = "deep_ptune"

            class DummyModel(nn.Module, PTuneMixin):
                def __init__(self):
                    super().__init__()
                    self.config = MockConfig()
                    self.word_embeddings = nn.Embedding(100, 10)
                    self.init_prompts(self.config)

            model = DummyModel()
            prompts, interm = model.get_prompt(2)

            assert prompts.shape == torch.Size([2, 5, 10])
            assert interm.shape == torch.Size([3, 2, 5, 10])
