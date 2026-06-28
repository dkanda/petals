import os
import sys

# Ensure torch is imported BEFORE patching sys.modules to avoid PyTorch sefaults
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from unittest import mock

# Mocking heavy dependencies that may be missing in test environments like hivemind
mocks = {
    'hivemind': mock.MagicMock(),
    'hivemind.compression': mock.MagicMock(),
    'hivemind.moe': mock.MagicMock(),
    'hivemind.moe.client': mock.MagicMock(),
    'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
    'hivemind.moe.server': mock.MagicMock(),
    'hivemind.moe.server.module_backend': mock.MagicMock(),
    'hivemind.moe.server.connection_handler': mock.MagicMock(),
    'hivemind.moe.expert_uid': mock.MagicMock(),
    'hivemind.p2p': mock.MagicMock(),
    'hivemind.p2p.p2p_daemon_bindings': mock.MagicMock(),
    'hivemind.p2p.p2p_daemon_bindings.control': mock.MagicMock(),
    'hivemind.p2p.p2p_daemon': mock.MagicMock(),
    'hivemind.dht': mock.MagicMock(),
    'hivemind.dht.node': mock.MagicMock(),
    'hivemind.proto': mock.MagicMock(),
    'hivemind.proto.runtime_pb2': mock.MagicMock(),
    'hivemind.utils': mock.MagicMock(),
    'hivemind.utils.asyncio': mock.MagicMock(),
    'hivemind.utils.logging': mock.MagicMock(),
    'hivemind.utils.mpfuture': mock.MagicMock(),
    'hivemind.utils.streaming': mock.MagicMock(),
    'hivemind.utils.nested': mock.MagicMock(),
    'hivemind.utils.tensor_descr': mock.MagicMock(),
    'hivemind.compression.serialization': mock.MagicMock(),
    'tensor_parallel': mock.MagicMock(),
    'tensor_parallel.slicing_configs': mock.MagicMock(),
    'tensor_parallel.tensor_parallel': mock.MagicMock(),
}

def test_ptune_intermediate_prompts_shape():
    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin
        from petals.utils.misc import DUMMY

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                # Mock word_embeddings to provide weight.device and weight.dtype
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=4,
            tuning_mode="deep_ptune",
            pre_seq_len=8
        )

        model = DummyModel(config)
        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
        assert intermediate_prompts.shape == (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
