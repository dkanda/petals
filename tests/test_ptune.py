import sys
import unittest
from unittest import mock

import torch
import torch.nn as nn
from transformers import PretrainedConfig

# Mock heavy dependencies like hivemind that fail to load locally in tests
mock_hivemind = mock.MagicMock()
mock_tensor_parallel = mock.MagicMock()

mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
mock_hivemind.p2p.p2p_daemon_bindings.control.MAX_UNARY_PAYLOAD_SIZE = 1000000

mocks = {
    'hivemind': mock_hivemind,
    'hivemind.compression': mock_hivemind.compression,
    'hivemind.compression.serialization': mock_hivemind.compression.serialization,
    'hivemind.dht': mock_hivemind.dht,
    'hivemind.dht.node': mock_hivemind.dht.node,
    'hivemind.moe': mock_hivemind.moe,
    'hivemind.moe.client': mock_hivemind.moe.client,
    'hivemind.moe.client.remote_expert_worker': mock_hivemind.moe.client.remote_expert_worker,
    'hivemind.moe.expert_uid': mock_hivemind.moe.expert_uid,
    'hivemind.moe.server': mock_hivemind.moe.server,
    'hivemind.moe.server.connection_handler': mock_hivemind.moe.server.connection_handler,
    'hivemind.moe.server.module_backend': mock_hivemind.moe.server.module_backend,
    'hivemind.p2p': mock_hivemind.p2p,
    'hivemind.p2p.p2p_daemon': mock_hivemind.p2p.p2p_daemon,
    'hivemind.p2p.p2p_daemon_bindings': mock_hivemind.p2p.p2p_daemon_bindings,
    'hivemind.p2p.p2p_daemon_bindings.control': mock_hivemind.p2p.p2p_daemon_bindings.control,
    'hivemind.p2p.p2p_daemon_bindings.datastructures': mock_hivemind.p2p.p2p_daemon_bindings.datastructures,
    'hivemind.proto': mock_hivemind.proto,
    'hivemind.proto.runtime_pb2': mock_hivemind.proto.runtime_pb2,
    'hivemind.utils': mock_hivemind.utils,
    'hivemind.utils.asyncio': mock_hivemind.utils.asyncio,
    'hivemind.utils.logging': mock_hivemind.utils.logging,
    'hivemind.utils.mpfuture': mock_hivemind.utils.mpfuture,
    'hivemind.utils.nested': mock_hivemind.utils.nested,
    'hivemind.utils.streaming': mock_hivemind.utils.streaming,
    'hivemind.utils.tensor_descr': mock_hivemind.utils.tensor_descr,
    'tensor_parallel': mock_tensor_parallel,
    'tensor_parallel.slicing_configs': mock_tensor_parallel.slicing_configs,
    'tensor_parallel.tensor_parallel': mock_tensor_parallel.tensor_parallel,
}

class TestPTuneMixin(unittest.TestCase):
    def test_deep_ptune_intermediate_prompts_shape(self):
        # We must patch sys.modules before importing PTuneMixin so that local petals modules
        # load correctly without hitting MissingDependency errors for hivemind
        with mock.patch.dict('sys.modules', mocks):
            from petals.client.ptune import PTuneMixin

            class DummyModel(nn.Module, PTuneMixin):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    # Initialize word_embeddings before calling init_prompts
                    # because get_prompt accesses self.word_embeddings.weight
                    self.word_embeddings = nn.Embedding(100, config.hidden_size)
                    self.init_prompts(config)

            config = PretrainedConfig(
                tuning_mode="deep_ptune",
                pre_seq_len=5,
                hidden_size=16,
                num_hidden_layers=4
            )

            model = DummyModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            # Layer 0 prompts shape
            self.assertEqual(prompts.shape, torch.Size([2, 5, 16]))
            # Subsequent layers shape: (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
            self.assertEqual(intermediate_prompts.shape, torch.Size([3, 2, 5, 16]))
