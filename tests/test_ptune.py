import os
import sys
import unittest
import torch
import torch.nn as nn
from unittest import mock

# Ensure the local src is on the path so we test the local package, not any installed version
sys.path.insert(0, os.path.abspath('src'))

class MockWordEmbeddings:
    def __init__(self):
        self.weight = torch.empty(0)

class MockConfig:
    def __init__(self):
        self.num_hidden_layers = 4
        self.hidden_size = 8
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 2

# We need to mock a large number of deep dependencies from hivemind and tensor_parallel
# since they are not available in the testing environment and are not needed for this pure logic test.
mocks = {
    'hivemind': mock.MagicMock(),
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
    'hivemind.compression': mock.MagicMock(),
    'hivemind.compression.serialization': mock.MagicMock(),
    'tensor_parallel': mock.MagicMock(),
    'tensor_parallel.slicing_configs': mock.MagicMock(),
    'tensor_parallel.tensor_parallel': mock.MagicMock(),
    'petals.utils.misc': mock.MagicMock(),
}
mocks['petals.utils.misc'].DUMMY = torch.empty(0)
mocks['hivemind.p2p.p2p_daemon_bindings.control'].DEFAULT_MAX_MSG_SIZE = 1000000
mocks['hivemind.p2p.p2p_daemon'].DEFAULT_MAX_MSG_SIZE = 1000000

# Bypass petals dependency version assertions just in case
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

class TestPTuneMixin(unittest.TestCase):
    @mock.patch.dict('sys.modules', mocks)
    @mock.patch('petals.client.ptune.force_non_empty_weights', return_value=mock.MagicMock())
    def test_ptune_mixin_prompts_shape(self, mock_force_non_empty_weights):
        # Import the module under test only after mocks are applied
        from petals.client.ptune import PTuneMixin

        class MockedPTuneMixin(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = MockWordEmbeddings()
                self.init_prompts(config)

        config = MockConfig()
        mixin = MockedPTuneMixin(config)

        batch_size = 3
        prompts, intermediate_prompts = mixin.get_prompt(batch_size=batch_size)

        # Verify shapes
        # prompts should be of shape (batch_size, pre_seq_len, hidden_size)
        self.assertEqual(prompts.shape, torch.Size([batch_size, config.pre_seq_len, config.hidden_size]))

        # intermediate_prompts should be of shape (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
        self.assertEqual(
            intermediate_prompts.shape,
            torch.Size([config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size])
        )

if __name__ == '__main__':
    unittest.main()
