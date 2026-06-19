import sys
import os
import unittest
from unittest import mock
import torch
from transformers import PretrainedConfig

sys.path.insert(0, os.path.abspath('src'))

# Deep mocking of hivemind to prevent ModuleNotFoundError when importing petals modules
mock_hivemind = mock.MagicMock()
mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
mock_hivemind.p2p.p2p_daemon_bindings.control.MAX_UNARY_PAYLOAD_SIZE = 1000000
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
    'hivemind.p2p.p2p_daemon': mock_hivemind,
    'hivemind.p2p.p2p_daemon_bindings': mock_hivemind,
    'hivemind.p2p.p2p_daemon_bindings.control': mock_hivemind,
    'hivemind.p2p.p2p_daemon_bindings.datastructures': mock_hivemind,
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
    'tensor_parallel.tensor_parallel': mock.MagicMock()
}

with mock.patch.dict('sys.modules', mocks):
    from petals.client.ptune import PTuneMixin
    from petals.utils.misc import DUMMY

class MockConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = kwargs.get('tuning_mode')
        self.pre_seq_len = kwargs.get('pre_seq_len')
        self.hidden_size = kwargs.get('hidden_size')
        self.num_hidden_layers = kwargs.get('num_hidden_layers')

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = mock.MagicMock()
        self.word_embeddings.weight.device = torch.device('cpu')
        self.word_embeddings.weight.dtype = torch.float32
        self.init_prompts(config)

class TestPTuneMixin(unittest.TestCase):
    def test_ptune_shapes(self):
        config = MockConfig(tuning_mode='ptune', pre_seq_len=5, hidden_size=64, num_hidden_layers=10)
        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        # Verify layer 0 prompts shape: (batch_size, pre_seq_len, hidden_size)
        self.assertEqual(prompts.shape, (2, 5, 64))

        # In standard ptune, intermediate_prompts should be DUMMY
        self.assertIs(intermediate_prompts, DUMMY)

    def test_deep_ptune_shapes(self):
        config = MockConfig(tuning_mode='deep_ptune', pre_seq_len=5, hidden_size=64, num_hidden_layers=10)
        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        # Verify layer 0 prompts shape: (batch_size, pre_seq_len, hidden_size)
        self.assertEqual(prompts.shape, (2, 5, 64))

        # In deep_ptune, intermediate_prompts should be computed and properly sized
        # The expected shape is (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
        self.assertEqual(intermediate_prompts.shape, (9, 2, 5, 64))

if __name__ == '__main__':
    unittest.main()
