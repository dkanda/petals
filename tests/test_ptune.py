import os
import sys
import unittest
from unittest import mock
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('src'))

# Mocks
mock_hivemind = mock.MagicMock()
mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
mock_hivemind.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000

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
    'tensor_parallel': mock.MagicMock(),
    'tensor_parallel.slicing_configs': mock.MagicMock(),
    'tensor_parallel.tensor_parallel': mock.MagicMock(),
}

with mock.patch.dict('sys.modules', mocks):
    from transformers import PretrainedConfig
    from petals.client.ptune import PTuneMixin, PTuneConfig

class DummyWordEmbeddings(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, hidden_size, dtype=torch.float32))

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = DummyWordEmbeddings(config.hidden_size)
        self.init_prompts(config)

class TestPTune(unittest.TestCase):
    def test_ptune(self):
        config = PretrainedConfig()
        config.hidden_size = 16
        config.num_hidden_layers = 4
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 5

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        self.assertEqual(prompts.shape, torch.Size([2, 5, 16]))
        self.assertEqual(intermediate_prompts.shape, torch.Size([3, 2, 5, 16]))

if __name__ == '__main__':
    unittest.main()
