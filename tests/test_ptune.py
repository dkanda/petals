import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.abspath('src'))

import torch
import torch.nn as nn
from transformers import PretrainedConfig

def test_ptune_intermediate_prompts_shape():
    mock_hivemind = mock.MagicMock()
    mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
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

    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin
        from petals.utils.misc import DUMMY

        class MockConfig(PretrainedConfig):
            def __init__(self, tuning_mode=None, pre_seq_len=0, hidden_size=64, num_hidden_layers=4, **kwargs):
                super().__init__(**kwargs)
                self.tuning_mode = tuning_mode
                self.pre_seq_len = pre_seq_len
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers

        class MockModel(PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = MockConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=64, num_hidden_layers=4)
        model = MockModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 64])
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 64])

if __name__ == '__main__':
    test_ptune_intermediate_prompts_shape()
    print("Test passed successfully.")
