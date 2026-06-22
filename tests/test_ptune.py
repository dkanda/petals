import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.abspath('src'))

import torch
import torch.nn as nn

# Mocks to avoid missing dependencies in local environment
mock_hivemind = mock.MagicMock()
mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
mock_hivemind.p2p.p2p_daemon_bindings.control.MAX_UNARY_PAYLOAD_SIZE = 1000000
mock_hivemind.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000

mocks = {
    'hivemind': mock_hivemind,
    'hivemind.moe': mock.MagicMock(),
    'hivemind.moe.client': mock.MagicMock(),
    'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
    'hivemind.moe.server': mock.MagicMock(),
    'hivemind.moe.server.module_backend': mock.MagicMock(),
    'hivemind.moe.server.connection_handler': mock.MagicMock(),
    'hivemind.moe.expert_uid': mock.MagicMock(),
    'hivemind.p2p': mock_hivemind.p2p,
    'hivemind.p2p.p2p_daemon_bindings': mock_hivemind.p2p.p2p_daemon_bindings,
    'hivemind.p2p.p2p_daemon_bindings.control': mock_hivemind.p2p.p2p_daemon_bindings.control,
    'hivemind.p2p.p2p_daemon': mock_hivemind.p2p.p2p_daemon,
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
    'tensor_parallel.slicing_configs': mock.MagicMock(),
    'tensor_parallel.tensor_parallel': mock.MagicMock(),
    'tensor_parallel': mock.MagicMock(),
}

def test_ptune_intermediate_prompt_size():
    with mock.patch.dict('sys.modules', mocks):
        from transformers import PretrainedConfig
        from petals.client.ptune import PTuneMixin

        class TestPTune(PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        class Config(PretrainedConfig):
            def __init__(self, tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=8, num_hidden_layers=3, **kwargs):
                super().__init__(**kwargs)
                self.tuning_mode = tuning_mode
                self.pre_seq_len = pre_seq_len
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers

        config = Config()
        model = TestPTune(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 8]), "prompts shape does not match expected."
        # expected intermediate shape: (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
        assert intermediate_prompts.shape == torch.Size([2, 2, 5, 8]), "intermediate_prompts shape does not match expected."
