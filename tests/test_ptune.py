import sys
from unittest import mock
import torch
import torch.nn as nn

def test_ptune_mixin():
    # Need to mock heavy dependencies before importing petals components
    mock_hivemind = mock.MagicMock()
    mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
    mock_hivemind.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000
    mock_hivemind.utils.tensor_descr.MAX_UNARY_PAYLOAD_SIZE = 1000000

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
        'hivemind.utils.tensor_descr': mock_hivemind.utils.tensor_descr,
        'hivemind.compression': mock.MagicMock(),
        'hivemind.compression.serialization': mock.MagicMock(),
        'tensor_parallel': mock.MagicMock(),
        'tensor_parallel.slicing_configs': mock.MagicMock(),
        'tensor_parallel.tensor_parallel': mock.MagicMock(),
        'petals.utils.misc': mock.MagicMock(),
    }
    mocks['petals.utils.misc'].DUMMY = torch.empty(0)

    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin

        class DummyConfig:
            tuning_mode = "deep_ptune"
            pre_seq_len = 10
            hidden_size = 64
            num_hidden_layers = 4

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        config = DummyConfig()
        model = DummyModel(config)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        assert prompts.shape == (2, 10, 64)
        # Should be num_hidden_layers - 1
        assert intermediate_prompts.shape == (3, 2, 10, 64)

if __name__ == "__main__":
    test_ptune_mixin()
