import sys
import os
import unittest
import unittest.mock as mock

# Ensure src is in PYTHONPATH
sys.path.insert(0, os.path.abspath('src'))

# Mock hivemind and tensor_parallel before importing
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
    'hivemind.utils.logging': mock.MagicMock(),
    'hivemind.utils.asyncio': mock.MagicMock(),
    'hivemind.utils.mpfuture': mock.MagicMock(),
    'hivemind.utils.streaming': mock.MagicMock(),
    'hivemind.utils.nested': mock.MagicMock(),
    'hivemind.utils.tensor_descr': mock.MagicMock(),
    'hivemind.compression': mock.MagicMock(),
    'hivemind.compression.serialization': mock.MagicMock(),
    'tensor_parallel': mock.MagicMock(),
    'tensor_parallel.slicing_configs': mock.MagicMock(),
    'tensor_parallel.tensor_parallel': mock.MagicMock(),
}

mocks['hivemind.p2p.p2p_daemon_bindings.control'].DEFAULT_MAX_MSG_SIZE = 1000000
mocks['hivemind.p2p.p2p_daemon_bindings.control'].MAX_UNARY_PAYLOAD_SIZE = 1000000
mocks['hivemind.p2p.p2p_daemon'].DEFAULT_MAX_MSG_SIZE = 1000000

class TestPTuneShape(unittest.TestCase):
    def test_deep_ptune_shape(self):
        with mock.patch.dict('sys.modules', mocks):
            from petals.client.ptune import PTuneMixin, PTuneConfig
            from transformers import PretrainedConfig
            import torch
            import torch.nn as nn

            class DummyModel(nn.Module, PTuneMixin):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            config = PretrainedConfig(
                hidden_size=16,
                num_hidden_layers=4,
                tuning_mode="deep_ptune",
                pre_seq_len=5
            )

            model = DummyModel(config)
            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

            self.assertEqual(prompts.shape, (batch_size, config.pre_seq_len, config.hidden_size))
            # Shape should be (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
            self.assertEqual(intermediate_prompts.shape, (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size))
