import os
import sys
import torch
import torch.nn as nn
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

mock_hivemind = mock.MagicMock()
mock_tensor_parallel = mock.MagicMock()

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
    'hivemind.utils.logging': mock_hivemind.utils.logging,
    'hivemind.utils.mpfuture': mock_hivemind.utils.mpfuture,
    'hivemind.utils.streaming': mock_hivemind.utils.streaming,
    'hivemind.utils.nested': mock_hivemind.utils.nested,
    'hivemind.utils.tensor_descr': mock_hivemind.utils.tensor_descr,
    'hivemind.compression': mock_hivemind.compression,
    'hivemind.compression.serialization': mock_hivemind.compression.serialization,
    'tensor_parallel': mock_tensor_parallel,
    'tensor_parallel.slicing_configs': mock_tensor_parallel.slicing_configs,
    'tensor_parallel.tensor_parallel': mock_tensor_parallel.tensor_parallel,
}

mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
mock_hivemind.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000


def test_ptune_intermediate_prompt_shape():
    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin, PTuneConfig
        from transformers import PretrainedConfig

        class DummyConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.tuning_mode = "deep_ptune"
                self.pre_seq_len = 5
                self.hidden_size = 10
                self.num_hidden_layers = 4

        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32

        model = DummyModel(DummyConfig())
        model.init_prompts(model.config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        # Original logic used num_hidden_layers. It should now be num_hidden_layers - 1.
        # Shape should be [num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size]
        # In our case: [4 - 1, 2, 5, 10] = [3, 2, 5, 10]
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 10])
        assert model.intermediate_prompt_embeddings.weight.shape == torch.Size([5, 3 * 10])
