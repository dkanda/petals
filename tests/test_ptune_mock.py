import sys
import os
import torch
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

# Mock hivemind to bypass missing dependencies on CI/local
hivemind_mock = mock.MagicMock()
mocks = {
    'hivemind': hivemind_mock,
    'hivemind.compression': hivemind_mock,
    'hivemind.compression.serialization': hivemind_mock,
    'hivemind.dht': hivemind_mock,
    'hivemind.dht.node': hivemind_mock,
    'hivemind.moe': hivemind_mock,
    'hivemind.moe.client': hivemind_mock,
    'hivemind.moe.client.remote_expert_worker': hivemind_mock,
    'hivemind.moe.expert_uid': hivemind_mock,
    'hivemind.moe.server': hivemind_mock,
    'hivemind.moe.server.connection_handler': hivemind_mock,
    'hivemind.moe.server.layers': hivemind_mock,
    'hivemind.moe.server.module_backend': hivemind_mock,
    'hivemind.moe.server.runtime': hivemind_mock,
    'hivemind.p2p': hivemind_mock,
    'hivemind.p2p.p2p_daemon': hivemind_mock,
    'hivemind.p2p.p2p_daemon_bindings': hivemind_mock,
    'hivemind.p2p.p2p_daemon_bindings.control': hivemind_mock,
    'hivemind.proto': hivemind_mock,
    'hivemind.proto.dht_pb2': hivemind_mock,
    'hivemind.proto.runtime_pb2': hivemind_mock,
    'hivemind.utils': hivemind_mock,
    'hivemind.utils.asyncio': hivemind_mock,
    'hivemind.utils.logging': hivemind_mock,
    'hivemind.utils.mpfuture': hivemind_mock,
    'hivemind.utils.nested': hivemind_mock,
    'hivemind.utils.networking': hivemind_mock,
    'hivemind.utils.streaming': hivemind_mock,
    'hivemind.utils.tensor_descr': hivemind_mock,
    'tensor_parallel': hivemind_mock,
    'tensor_parallel.tensor_parallel': hivemind_mock,
    'tensor_parallel.slicing_configs': hivemind_mock,
}

class TestPTune(unittest.TestCase):
    @mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True)
    def test_deep_ptune_intermediate_shape(self, _):
        with mock.patch.dict('sys.modules', mocks):
            from transformers import PretrainedConfig
            from petals.client.ptune import PTuneMixin

            class DummyModel(PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            config = PretrainedConfig(
                tuning_mode="deep_ptune",
                pre_seq_len=5,
                hidden_size=16,
                num_hidden_layers=4,
            )

            model = DummyModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            # Layer 0 prompt has shape [batch_size, pre_seq_len, hidden_size]
            self.assertEqual(prompts.shape, torch.Size([2, 5, 16]))
            # Layer > 0 intermediate prompts have shape [num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size]
            self.assertEqual(intermediate_prompts.shape, torch.Size([3, 2, 5, 16]))

if __name__ == '__main__':
    unittest.main()
