import sys
import os
sys.path.insert(0, os.path.abspath('src'))
import torch
from unittest import mock

def test_ptune_intermediate_prompts_shape():
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

    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin
        from transformers import PretrainedConfig
        import petals.utils.misc

        class TestPTune(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.Mock()
                self.word_embeddings.weight = mock.Mock(dtype=torch.float32, device=torch.device('cpu'))
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=4,
            hidden_size=8,
            num_hidden_layers=3
        )

        model = TestPTune(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        # In deep_ptune, intermediate_prompts should have shape (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
        assert intermediate_prompts.shape == (2, 2, 4, 8)