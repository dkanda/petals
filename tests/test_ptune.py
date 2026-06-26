import sys
import os
import torch
import torch.nn as nn
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_mixin_shapes():
    mock_hivemind = mock.MagicMock()
    mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
    mock_hivemind.p2p.p2p_daemon_bindings.control.MAX_UNARY_PAYLOAD_SIZE = 1000000
    mock_hivemind.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000

    mocks = {
        'hivemind': mock_hivemind,
        'hivemind.p2p': mock_hivemind,
        'hivemind.moe': mock_hivemind,
        'hivemind.moe.client': mock_hivemind,
        'hivemind.moe.client.remote_expert_worker': mock_hivemind,
        'hivemind.moe.server': mock_hivemind,
        'hivemind.moe.server.module_backend': mock_hivemind,
        'hivemind.moe.server.connection_handler': mock_hivemind,
        'hivemind.moe.expert_uid': mock_hivemind,
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
        'tensor_parallel.slicing_configs': mock.MagicMock(),
        'tensor_parallel.tensor_parallel': mock.MagicMock(),
        'tensor_parallel': mock.MagicMock()
    }

    with mock.patch.dict('sys.modules', mocks):
        from transformers import PretrainedConfig
        from petals.client.ptune import PTuneMixin
        from petals.utils.misc import DUMMY

        class MockModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=10
        )

        # Avoid init_empty_weights logic by patching register_parameter with original one directly
        with mock.patch('petals.client.ptune._original_register_parameter', nn.Module.register_parameter):
            model = MockModel(config)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 16]), f"Expected [2, 5, 16], got {prompts.shape}"
        assert intermediate_prompts.shape == torch.Size([9, 2, 5, 16]), f"Expected [9, 2, 5, 16], got {intermediate_prompts.shape}"

if __name__ == "__main__":
    test_ptune_mixin_shapes()
    print("Test passed successfully!")
