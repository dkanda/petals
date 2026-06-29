import os
import sys
from unittest import mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig

sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

def test_deep_ptune_shapes():
    mock_hivemind = mock.MagicMock()
    mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
    mock_hivemind.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000
    mock_hivemind.MAX_UNARY_PAYLOAD_SIZE = 1000000
    mock_tensor_parallel = mock.MagicMock()

    mocks = {
        'hivemind': mock_hivemind,
        'hivemind.moe.client.remote_expert_worker': mock_hivemind,
        'hivemind.moe.server.module_backend': mock_hivemind,
        'hivemind.moe.server.connection_handler': mock_hivemind,
        'hivemind.moe.expert_uid': mock_hivemind,
        'hivemind.p2p.p2p_daemon_bindings.control': mock_hivemind,
        'hivemind.p2p.p2p_daemon': mock_hivemind,
        'hivemind.dht.node': mock_hivemind,
        'hivemind.dht': mock_hivemind,
        'hivemind.proto.runtime_pb2': mock_hivemind,
        'hivemind.proto': mock_hivemind,
        'hivemind.utils.asyncio': mock_hivemind,
        'hivemind.utils.logging': mock_hivemind,
        'hivemind.utils.mpfuture': mock_hivemind,
        'hivemind.utils.streaming': mock_hivemind,
        'hivemind.utils.nested': mock_hivemind,
        'hivemind.utils.tensor_descr': mock_hivemind,
        'hivemind.compression.serialization': mock_hivemind,
        'tensor_parallel.slicing_configs': mock_tensor_parallel,
        'tensor_parallel.tensor_parallel': mock_tensor_parallel,
        'tensor_parallel': mock_tensor_parallel,
        'hivemind.p2p': mock_hivemind,
        'hivemind.moe.client.expert': mock_hivemind,
        'hivemind.moe': mock_hivemind,
        'hivemind.compression': mock_hivemind,
        'hivemind.utils': mock_hivemind,
        'hivemind.p2p.p2p_daemon_bindings': mock_hivemind,
    }

    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=4
        )

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert intermediate_prompts.shape == (3, 2, 5, 16)
