import sys
import os
import torch
import torch.nn as nn
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_intermediate_prompt_embeddings_shape():
    mock_hivemind = mock.MagicMock()
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
        'hivemind.p2p.p2p_daemon': mock.MagicMock(),
        'hivemind.p2p.p2p_daemon_bindings': mock.MagicMock(),
        'hivemind.p2p.p2p_daemon_bindings.control': mock.MagicMock(),
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
        sys.modules['hivemind'].p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
        sys.modules['hivemind'].p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000
        sys.modules['hivemind'].compression.serialization.MAX_UNARY_PAYLOAD_SIZE = 1000000

        from petals.client.ptune import PTuneMixin
        from petals.utils.misc import DUMMY

        class DummyModel(PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, 8)

        config = mock.MagicMock()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 5
        config.hidden_size = 8
        config.num_hidden_layers = 12

        model = DummyModel(config)

        # Avoid init_empty_weights logic bypassing bugs
        with mock.patch('petals.client.ptune.force_non_empty_weights'):
            model.init_prompts(config)

        # verify intermediate_prompt_embeddings shape
        expected_embedding_dim = (config.num_hidden_layers - 1) * config.hidden_size
        assert model.intermediate_prompt_embeddings.weight.shape == (config.pre_seq_len, expected_embedding_dim), \
            f"Expected shape ({config.pre_seq_len}, {expected_embedding_dim}), got {model.intermediate_prompt_embeddings.weight.shape}"

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

        # verify intermediate_prompts shape
        # [num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size]
        expected_shape = (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
        assert intermediate_prompts.shape == expected_shape, \
            f"Expected intermediate prompts shape {expected_shape}, got {intermediate_prompts.shape}"
