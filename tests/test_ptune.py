import os
import sys
import torch
import torch.nn as nn
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

from transformers import PretrainedConfig
from unittest import mock

# In an isolated testing environment, mock hivemind
# since it may not be present when tests are collected
try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    mock_hivemind = mock.MagicMock()
    mocks = {
        'hivemind': mock_hivemind,
        'hivemind.moe': mock_hivemind.moe,
        'hivemind.moe.client': mock_hivemind.moe.client,
        'hivemind.moe.client.remote_expert_worker': mock_hivemind.moe.client.remote_expert_worker,
        'hivemind.moe.server': mock_hivemind.moe.server,
        'hivemind.moe.server.module_backend': mock_hivemind.moe.server.module_backend,
        'hivemind.moe.server.connection_handler': mock_hivemind.moe.server.connection_handler,
        'hivemind.moe.expert_uid': mock_hivemind.moe.expert_uid,
        'hivemind.p2p': mock_hivemind.p2p,
        'hivemind.p2p.p2p_daemon_bindings': mock_hivemind.p2p.p2p_daemon_bindings,
        'hivemind.p2p.p2p_daemon_bindings.control': mock_hivemind.p2p.p2p_daemon_bindings.control,
        'hivemind.p2p.p2p_daemon': mock_hivemind.p2p.p2p_daemon,
        'hivemind.dht': mock_hivemind.dht,
        'hivemind.dht.node': mock_hivemind.dht.node,
        'hivemind.proto': mock_hivemind.proto,
        'hivemind.proto.runtime_pb2': mock_hivemind.proto.runtime_pb2,
        'hivemind.utils': mock_hivemind.utils,
        'hivemind.utils.asyncio': mock_hivemind.utils.asyncio,
        'hivemind.utils.logging': mock_hivemind.utils.logging,
        'hivemind.utils.mpfuture': mock_hivemind.utils.mpfuture,
        'hivemind.utils.streaming': mock_hivemind.utils.streaming,
        'hivemind.utils.nested': mock_hivemind.utils.nested,
        'hivemind.utils.tensor_descr': mock_hivemind.utils.tensor_descr,
        'hivemind.compression': mock_hivemind.compression,
        'hivemind.compression.serialization': mock_hivemind.compression.serialization,
        'tensor_parallel': mock.MagicMock(),
        'tensor_parallel.slicing_configs': mock.MagicMock(),
        'tensor_parallel.tensor_parallel': mock.MagicMock(),
    }
    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin

def test_ptune_deep_ptune():
    class MockModel(nn.Module, PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5
    config.hidden_size = 16
    config.num_hidden_layers = 4

    model = MockModel(config)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == torch.Size([2, 5, 16])
    assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16])
