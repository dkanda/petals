import sys
import unittest.mock as mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import pytest

def test_deep_ptune():
    mock_hivemind = mock.MagicMock()
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
        'hivemind.p2p.p2p_daemon': mock_hivemind,
        'hivemind.p2p.p2p_daemon_bindings': mock_hivemind,
        'hivemind.p2p.p2p_daemon_bindings.control': mock_hivemind,
        'hivemind.compression': mock_hivemind,
        'hivemind.compression.serialization': mock_hivemind,
        'hivemind.dht': mock_hivemind,
        'hivemind.dht.node': mock_hivemind,
        'hivemind.utils': mock_hivemind,
        'hivemind.utils.asyncio': mock_hivemind,
        'hivemind.utils.logging': mock_hivemind,
        'hivemind.utils.mpfuture': mock_hivemind,
        'hivemind.utils.streaming': mock_hivemind,
        'hivemind.utils.nested': mock_hivemind,
        'hivemind.utils.tensor_descr': mock_hivemind,
        'hivemind.proto': mock_hivemind,
        'hivemind.proto.runtime_pb2': mock_hivemind,
        'tensor_parallel': mock_hivemind,
        'tensor_parallel.slicing_configs': mock_hivemind,
        'tensor_parallel.tensor_parallel': mock_hivemind,
        'speedtest': mock_hivemind,
    }
    mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
    mock_hivemind.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000

    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin
        from petals.utils.misc import DUMMY

        class MockWordEmbeddings:
            def __init__(self):
                self.weight = torch.empty(1, dtype=torch.float32)

        class MockModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = MockWordEmbeddings()
                self.init_prompts(config)

        config = PretrainedConfig()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 5
        config.hidden_size = 10
        config.num_hidden_layers = 4

        model = MockModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 10])
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 10])
