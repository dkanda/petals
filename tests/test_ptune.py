import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import torch
import torch.nn as nn
from unittest import mock

# Mock hivemind and tensor_parallel explicitly
mock_hivemind = mock.MagicMock()
mock_hivemind.get_logger.return_value = mock.MagicMock()

mocks = {
    'hivemind': mock_hivemind,
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
}
mocks['hivemind.p2p.p2p_daemon_bindings.control'].DEFAULT_MAX_MSG_SIZE = 1000000
mocks['hivemind.p2p.p2p_daemon_bindings.control'].MAX_UNARY_PAYLOAD_SIZE = 1000000
mocks['hivemind.p2p.p2p_daemon'].DEFAULT_MAX_MSG_SIZE = 1000000

with mock.patch.dict('sys.modules', mocks):
    from petals.client.ptune import PTuneMixin, PTuneConfig
    from transformers import PretrainedConfig

class DummyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 10
        self.hidden_size = 128
        self.num_hidden_layers = 4

class MockPTuneModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = nn.Embedding(10, 128)
        self.init_prompts(config)

def test_ptune_mixin_shapes():
    config = DummyConfig()
    model = MockPTuneModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size)
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (
        config.num_hidden_layers - 1,
        batch_size,
        config.pre_seq_len,
        config.hidden_size,
    )

if __name__ == '__main__':
    test_ptune_mixin_shapes()
