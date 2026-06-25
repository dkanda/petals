import sys
import os
from unittest import mock

import torch
import torch.nn as nn
from transformers import PretrainedConfig

# Mock deep dependencies
mock_hivemind = mock.MagicMock()
mock_hivemind.p2p = mock.MagicMock()
mock_hivemind.p2p.p2p_daemon_bindings = mock.MagicMock()
mock_hivemind.p2p.p2p_daemon_bindings.control = mock.MagicMock()
mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
mock_hivemind.p2p.p2p_daemon = mock.MagicMock()
mock_hivemind.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000
mock_hivemind.compression = mock.MagicMock()
mock_hivemind.compression.serialization = mock.MagicMock()
mock_hivemind.moe = mock.MagicMock()
mock_hivemind.moe.client = mock.MagicMock()
mock_hivemind.moe.client.remote_expert_worker = mock.MagicMock()
mock_hivemind.moe.server = mock.MagicMock()
mock_hivemind.moe.server.module_backend = mock.MagicMock()
mock_hivemind.moe.server.connection_handler = mock.MagicMock()
mock_hivemind.moe.expert_uid = mock.MagicMock()
mock_hivemind.dht = mock.MagicMock()
mock_hivemind.dht.node = mock.MagicMock()
mock_hivemind.proto = mock.MagicMock()
mock_hivemind.proto.runtime_pb2 = mock.MagicMock()
mock_hivemind.utils = mock.MagicMock()
mock_hivemind.utils.asyncio = mock.MagicMock()
mock_hivemind.utils.logging = mock.MagicMock()
mock_hivemind.utils.mpfuture = mock.MagicMock()
mock_hivemind.utils.streaming = mock.MagicMock()
mock_hivemind.utils.nested = mock.MagicMock()
mock_hivemind.utils.tensor_descr = mock.MagicMock()

mock_tensor_parallel = mock.MagicMock()
mock_tensor_parallel.slicing_configs = mock.MagicMock()
mock_tensor_parallel.tensor_parallel = mock.MagicMock()

mocks = {
    'hivemind': mock_hivemind,
    'hivemind.p2p': mock_hivemind.p2p,
    'hivemind.p2p.P2P': mock_hivemind.p2p.P2P,
    'hivemind.p2p.p2p_daemon_bindings': mock_hivemind.p2p.p2p_daemon_bindings,
    'hivemind.p2p.p2p_daemon_bindings.control': mock_hivemind.p2p.p2p_daemon_bindings.control,
    'hivemind.p2p.p2p_daemon': mock_hivemind.p2p.p2p_daemon,
    'hivemind.compression': mock_hivemind.compression,
    'hivemind.compression.serialization': mock_hivemind.compression.serialization,
    'hivemind.moe': mock_hivemind.moe,
    'hivemind.moe.client': mock_hivemind.moe.client,
    'hivemind.moe.client.remote_expert_worker': mock_hivemind.moe.client.remote_expert_worker,
    'hivemind.moe.server': mock_hivemind.moe.server,
    'hivemind.moe.server.module_backend': mock_hivemind.moe.server.module_backend,
    'hivemind.moe.server.connection_handler': mock_hivemind.moe.server.connection_handler,
    'hivemind.moe.expert_uid': mock_hivemind.moe.expert_uid,
    'hivemind.dht': mock_hivemind.dht,
    'hivemind.dht.node': mock_hivemind.dht.node,
    'hivemind.proto': mock_hivemind.proto,
    'hivemind.proto.runtime_pb2': mock_hivemind.proto.runtime_pb2,
    'hivemind.utils': mock_hivemind.utils,
    'hivemind.utils.asyncio': mock_hivemind.utils.asyncio,
    'hivemind.utils.logging': mock_hivemind.utils.logging,
    'hivemind.utils.mpfuture': mock_hivemind.utils.mpfuture,
    'hivemind.utils.streaming': mock_hivemind.utils.streaming,
    'hivemind.utils.nested': mock_hivemind.utils.nested,
    'hivemind.utils.tensor_descr': mock_hivemind.utils.tensor_descr,
    'tensor_parallel': mock_tensor_parallel,
    'tensor_parallel.slicing_configs': mock_tensor_parallel.slicing_configs,
    'tensor_parallel.tensor_parallel': mock_tensor_parallel.tensor_parallel,
}

os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

def test_ptune_intermediate_prompt_embeddings():
    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        # deep_ptune mode check
        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=10,
            tuning_mode="deep_ptune",
            pre_seq_len=5
        )
        model = DummyModel(config)

        # Check initialized embedding weights shape
        assert model.prompt_embeddings.weight.shape == (5, 64)
        assert model.intermediate_prompt_embeddings.weight.shape == (5, 9 * 64)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert prompts.shape == (2, 5, 64)
        # Verify shape of intermediate_prompts returned
        assert intermediate_prompts.shape == (9, 2, 5, 64)

        # ptune mode check
        config_ptune = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=10,
            tuning_mode="ptune",
            pre_seq_len=5
        )
        model_ptune = DummyModel(config_ptune)

        assert model_ptune.prompt_embeddings.weight.shape == (5, 64)
        assert not hasattr(model_ptune, 'intermediate_prompt_embeddings')
