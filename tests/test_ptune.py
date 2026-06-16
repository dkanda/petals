import sys
import os
import unittest
from unittest import mock
import torch
from transformers import PretrainedConfig

sys.path.insert(0, os.path.abspath('src'))

# Mock hivemind and other deep dependencies to bypass installation issues in local environment
hivemind_mock = mock.MagicMock()
hivemind_mock.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
hivemind_mock.p2p.p2p_daemon_bindings.control.MAX_UNARY_PAYLOAD_SIZE = 1000000

mocks = {
    'hivemind': hivemind_mock,
    'hivemind.compression': hivemind_mock,
    'hivemind.compression.serialization': hivemind_mock,
    'hivemind.moe': hivemind_mock,
    'hivemind.moe.client': hivemind_mock,
    'hivemind.moe.client.remote_expert_worker': hivemind_mock,
    'hivemind.moe.expert_uid': hivemind_mock,
    'hivemind.moe.server': hivemind_mock,
    'hivemind.moe.server.connection_handler': hivemind_mock,
    'hivemind.moe.server.module_backend': hivemind_mock,
    'hivemind.p2p': hivemind_mock,
    'hivemind.p2p.p2p_daemon': hivemind_mock,
    'hivemind.p2p.p2p_daemon_bindings': hivemind_mock,
    'hivemind.p2p.p2p_daemon_bindings.control': hivemind_mock,
    'hivemind.utils': hivemind_mock,
    'hivemind.utils.tensor_descr': hivemind_mock,
    'hivemind.utils.logging': hivemind_mock,
    'hivemind.utils.asyncio': hivemind_mock,
    'hivemind.utils.streaming': hivemind_mock,
    'hivemind.utils.mpfuture': hivemind_mock,
    'hivemind.utils.nested': hivemind_mock,
    'hivemind.proto': hivemind_mock,
    'hivemind.proto.runtime_pb2': hivemind_mock,
    'hivemind.dht': hivemind_mock,
    'hivemind.dht.node': hivemind_mock,
    'tensor_parallel': mock.MagicMock(),
    'tensor_parallel.slicing': mock.MagicMock(),
    'tensor_parallel.slicing_configs': mock.MagicMock(),
    'tensor_parallel.tensor_parallel': mock.MagicMock(),
}

class TestPTuneMixin(unittest.TestCase):
    def test_deep_ptune_shape(self):
        with mock.patch.dict('sys.modules', mocks):
            from petals.client.ptune import PTuneMixin, PTuneConfig

            class DummyModel(PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    # Create dummy word_embeddings weight for device/dtype
                    self.word_embeddings = mock.MagicMock()
                    self.word_embeddings.weight = torch.empty(0, dtype=torch.float32, device='cpu')

            config = PretrainedConfig()
            config.tuning_mode = "deep_ptune"
            config.pre_seq_len = 16
            config.hidden_size = 64
            config.num_hidden_layers = 12

            model = DummyModel(config)
            model.init_prompts(config)

            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            self.assertEqual(prompts.shape, (batch_size, config.pre_seq_len, config.hidden_size))
            self.assertEqual(intermediate_prompts.shape, (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size))

if __name__ == '__main__':
    unittest.main()
