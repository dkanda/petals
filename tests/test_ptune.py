import sys
import os
import unittest
from unittest import mock
import torch

sys.path.insert(0, os.path.abspath('src'))

# Mock missing dependencies
hivemind_mock = mock.MagicMock()
hivemind_mock.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
hivemind_mock.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000
hivemind_mock.utils.nested.MAX_UNARY_PAYLOAD_SIZE = 1000000

tensor_parallel_mock = mock.MagicMock()

mocks = {
    'hivemind': hivemind_mock,
    'hivemind.p2p': hivemind_mock.p2p,
    'hivemind.utils': hivemind_mock.utils,
    'hivemind.utils.logging': mock.MagicMock(),
    'hivemind.utils.nested': hivemind_mock.utils.nested,
    'hivemind.utils.asyncio': mock.MagicMock(),
    'hivemind.utils.mpfuture': mock.MagicMock(),
    'hivemind.utils.streaming': mock.MagicMock(),
    'hivemind.utils.tensor_descr': mock.MagicMock(),
    'hivemind.dht': mock.MagicMock(),
    'hivemind.dht.node': mock.MagicMock(),
    'hivemind.moe': mock.MagicMock(),
    'hivemind.moe.client': mock.MagicMock(),
    'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
    'hivemind.moe.server': mock.MagicMock(),
    'hivemind.moe.server.module_backend': mock.MagicMock(),
    'hivemind.moe.server.connection_handler': mock.MagicMock(),
    'hivemind.moe.expert_uid': mock.MagicMock(),
    'hivemind.proto': mock.MagicMock(),
    'hivemind.proto.runtime_pb2': mock.MagicMock(),
    'hivemind.compression': mock.MagicMock(),
    'hivemind.compression.serialization': mock.MagicMock(),
    'hivemind.p2p.p2p_daemon_bindings': mock.MagicMock(),
    'hivemind.p2p.p2p_daemon_bindings.control': hivemind_mock.p2p.p2p_daemon_bindings.control,
    'hivemind.p2p.p2p_daemon': hivemind_mock.p2p.p2p_daemon,
    'tensor_parallel': tensor_parallel_mock,
    'tensor_parallel.slicing_configs': mock.MagicMock(),
    'tensor_parallel.tensor_parallel': mock.MagicMock(),
    'speedtest': mock.MagicMock(),
    'pydantic': mock.MagicMock(),
    'pydantic.v1': mock.MagicMock(),
    'fastapi': mock.MagicMock(),
    'starlette': mock.MagicMock(),
    'uvicorn': mock.MagicMock(),
    'websockets': mock.MagicMock(),
    'async_timeout': mock.MagicMock(),
    'dijkstar': mock.MagicMock(),
}

with mock.patch.dict('sys.modules', mocks):
    from petals.client.ptune import PTuneMixin
    from transformers import PretrainedConfig

class DummyWordEmbeddings:
    weight = torch.zeros(1, 16, dtype=torch.float16)

class ModelTest(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = DummyWordEmbeddings()
        self.init_prompts(config)

class TestPTuneMixin(unittest.TestCase):
    def test_get_prompt_deep_ptune(self):
        config = PretrainedConfig(
            hidden_size=16,
            num_hidden_layers=4,
            tuning_mode="deep_ptune",
            pre_seq_len=3
        )

        model = ModelTest(config)
        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        # In deep_ptune, the intermediate prompts should have shape:
        # (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
        self.assertEqual(intermediate_prompts.shape, (3, batch_size, 3, 16))

        # The primary prompt embeddings should have shape:
        # (batch_size, pre_seq_len, hidden_size)
        self.assertEqual(prompts.shape, (batch_size, 3, 16))
