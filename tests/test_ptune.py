import sys
from unittest import mock
import os

sys.path.insert(0, os.path.abspath('src'))

mock_hivemind = mock.MagicMock()
mock_tensor_parallel = mock.MagicMock()

# Hivemind nested mocks
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
    'tensor_parallel': mock_tensor_parallel,
    'tensor_parallel.slicing_configs': mock_tensor_parallel,
    'tensor_parallel.tensor_parallel': mock_tensor_parallel,
}

with mock.patch.dict('sys.modules', mocks):
    from petals.client.ptune import PTuneMixin, PTuneConfig
    from transformers import PretrainedConfig
    import torch
    import torch.nn as nn

    class DummyModel(PTuneMixin, nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(10, config.hidden_size)
            self.init_prompts(config)

def test_deep_ptune_dimensions():
    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5
    config.hidden_size = 8
    config.num_hidden_layers = 4

    model = DummyModel(config)

    # Check intermediate embedding weight shape
    # It should be pre_seq_len x ((num_hidden_layers - 1) * hidden_size)
    assert model.intermediate_prompt_embeddings.weight.shape == (5, 3 * 8)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    # Prompts for layer 0
    assert prompts.shape == (2, 5, 8)

    # Intermediate prompts shape
    # (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (3, 2, 5, 8)
