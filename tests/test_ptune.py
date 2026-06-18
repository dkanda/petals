import sys
import os
import pytest
import torch
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_intermediate_prompts_shape():
    mock_hivemind = mock.MagicMock()
    mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
    mock_hivemind.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000

    mocks = {
        'hivemind': mock_hivemind,
        'hivemind.moe': mock_hivemind.moe,
        'hivemind.moe.client': mock_hivemind.moe.client,
        'hivemind.moe.client.remote_expert_worker': mock_hivemind.moe.client.remote_expert_worker,
        'hivemind.moe.server': mock_hivemind.moe.server,
        'hivemind.moe.server.module_backend': mock_hivemind.moe.server.module_backend,
        'hivemind.moe.server.connection_handler': mock_hivemind.moe.server.connection_handler,
        'hivemind.moe.expert_uid': mock_hivemind.moe.expert_uid,
        'hivemind.p2p': mock_hivemind.p2p,
        'hivemind.p2p.p2p_daemon_bindings': mock_hivemind.p2p.p2p_daemon_bindings,
        'hivemind.p2p.p2p_daemon_bindings.control': mock_hivemind.p2p.p2p_daemon_bindings.control,
        'hivemind.p2p.p2p_daemon': mock_hivemind.p2p.p2p_daemon,
        'hivemind.dht': mock_hivemind.dht,
        'hivemind.dht.node': mock_hivemind.dht.node,
        'hivemind.proto': mock_hivemind.proto,
        'hivemind.proto.runtime_pb2': mock_hivemind.proto.runtime_pb2,
        'hivemind.utils': mock_hivemind.utils,
        'hivemind.utils.asyncio': mock_hivemind.utils.asyncio,
        'hivemind.utils.mpfuture': mock_hivemind.utils.mpfuture,
        'hivemind.utils.streaming': mock_hivemind.utils.streaming,
        'hivemind.utils.nested': mock_hivemind.utils.nested,
        'hivemind.utils.tensor_descr': mock_hivemind.utils.tensor_descr,
        'hivemind.utils.logging': mock_hivemind.utils.logging,
        'hivemind.compression': mock_hivemind.compression,
        'hivemind.compression.serialization': mock_hivemind.compression.serialization,
        'tensor_parallel': mock.MagicMock(),
        'tensor_parallel.slicing_configs': mock.MagicMock(),
        'tensor_parallel.tensor_parallel': mock.MagicMock(),
    }

    with mock.patch.dict('sys.modules', mocks):
        with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
            from petals.client.ptune import PTuneMixin

            class Config:
                def __init__(self, tuning_mode, pre_seq_len, hidden_size, num_hidden_layers):
                    self.tuning_mode = tuning_mode
                    self.pre_seq_len = pre_seq_len
                    self.hidden_size = hidden_size
                    self.num_hidden_layers = num_hidden_layers

            class TestModel(PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.init_prompts(config)
                    self.word_embeddings = mock.MagicMock()
                    self.word_embeddings.weight = torch.empty(0, dtype=torch.float32)

            config = Config("deep_ptune", 10, 32, 5)
            model = TestModel(config)

            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            # Assertions
            expected_shape = (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
            assert intermediate_prompts.shape == expected_shape, f"Expected {expected_shape}, got {intermediate_prompts.shape}"
