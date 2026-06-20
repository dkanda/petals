import sys
import os
sys.path.insert(0, os.path.abspath('src'))

from unittest import mock
import torch

class MockModule(mock.MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return mock.MagicMock()

# Mock deep dependencies that fail to resolve in isolated environment
mocks = {
    'hivemind': MockModule(),
    'hivemind.p2p': MockModule(),
    'hivemind.moe': MockModule(),
    'hivemind.moe.client': MockModule(),
    'hivemind.moe.server': MockModule(),
    'hivemind.moe.client.remote_expert_worker': MockModule(),
    'hivemind.moe.server.module_backend': MockModule(),
    'hivemind.moe.server.connection_handler': MockModule(),
    'hivemind.moe.expert_uid': MockModule(),
    'hivemind.p2p.p2p_daemon_bindings': MockModule(),
    'hivemind.p2p.p2p_daemon_bindings.control': MockModule(),
    'hivemind.p2p.p2p_daemon': MockModule(),
    'hivemind.dht': MockModule(),
    'hivemind.dht.node': MockModule(),
    'hivemind.proto': MockModule(),
    'hivemind.proto.runtime_pb2': MockModule(),
    'hivemind.utils': MockModule(),
    'hivemind.utils.asyncio': MockModule(),
    'hivemind.utils.logging': MockModule(),
    'hivemind.utils.mpfuture': MockModule(),
    'hivemind.utils.streaming': MockModule(),
    'hivemind.utils.nested': MockModule(),
    'hivemind.utils.tensor_descr': MockModule(),
    'hivemind.compression': MockModule(),
    'hivemind.compression.serialization': MockModule(),
    'tensor_parallel': MockModule(),
    'tensor_parallel.slicing_configs': MockModule(),
    'tensor_parallel.tensor_parallel': MockModule(),
}

os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

with mock.patch.dict('sys.modules', mocks):
    from petals.client.ptune import PTuneMixin
    from petals.utils.misc import DUMMY

class DummyConfig:
    def __init__(self):
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 10
        self.hidden_size = 64
        self.num_hidden_layers = 12

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = mock.MagicMock()
        self.word_embeddings.weight.device = torch.device('cpu')
        self.word_embeddings.weight.dtype = torch.float32
        self.init_prompts(config)

def test_ptunemixin_intermediate_prompts_shape():
    """
    Tests that the intermediate prompt embeddings created by PTuneMixin
    in 'deep_ptune' mode have the correct shape:
    (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size).
    """
    config = DummyConfig()
    model = DummyModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    # Prompt shape should be (batch_size, pre_seq_len, hidden_size)
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # Intermediate prompts shape should be (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    expected_shape = (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == expected_shape, f"Expected {expected_shape}, got {intermediate_prompts.shape}"
