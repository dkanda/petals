import os
import sys
import torch
import torch.nn as nn
from unittest import mock
from transformers import PretrainedConfig

def test_ptune_intermediate_prompts_shape():
    mock_hivemind = mock.MagicMock()
    mock_tensor_parallel = mock.MagicMock()

    mocks = {
        'hivemind': mock_hivemind,
        'hivemind.p2p': mock.MagicMock(),
        'hivemind.p2p.p2p_daemon_bindings.control': mock.MagicMock(),
        'hivemind.p2p.p2p_daemon': mock.MagicMock(),
        'hivemind.dht': mock.MagicMock(),
        'hivemind.moe': mock.MagicMock(),
        'hivemind.moe.client': mock.MagicMock(),
        'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
        'hivemind.moe.server': mock.MagicMock(),
        'hivemind.moe.server.module_backend': mock.MagicMock(),
        'hivemind.moe.server.connection_handler': mock.MagicMock(),
        'hivemind.moe.expert_uid': mock.MagicMock(),
        'hivemind.utils': mock.MagicMock(),
        'hivemind.utils.asyncio': mock.MagicMock(),
        'hivemind.utils.logging': mock.MagicMock(),
        'hivemind.utils.mpfuture': mock.MagicMock(),
        'hivemind.utils.streaming': mock.MagicMock(),
        'hivemind.utils.nested': mock.MagicMock(),
        'hivemind.utils.tensor_descr': mock.MagicMock(),
        'hivemind.compression': mock.MagicMock(),
        'hivemind.compression.serialization': mock.MagicMock(),
        'hivemind.proto': mock.MagicMock(),
        'hivemind.proto.runtime_pb2': mock.MagicMock(),
        'hivemind.dht.node': mock.MagicMock(),
        'tensor_parallel': mock_tensor_parallel,
        'tensor_parallel.slicing_configs': mock.MagicMock(),
        'tensor_parallel.tensor_parallel': mock.MagicMock(),
    }

    # Fix some constants used when petals gets imported
    mocks['hivemind.p2p.p2p_daemon_bindings.control'].DEFAULT_MAX_MSG_SIZE = 1000000
    mocks['hivemind.p2p.p2p_daemon_bindings.control'].MAX_UNARY_PAYLOAD_SIZE = 1000000
    mocks['hivemind.p2p.p2p_daemon'].DEFAULT_MAX_MSG_SIZE = 1000000

    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin

        class DummyModel(PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            pre_seq_len=4,
            tuning_mode="deep_ptune",
            hidden_size=16,
            num_hidden_layers=3
        )

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 4, 16)
        assert intermediate_prompts.shape == (2, 2, 4, 16)
