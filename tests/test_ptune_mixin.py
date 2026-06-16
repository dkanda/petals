import os
import sys
from unittest import mock
import torch

sys.path.insert(0, os.path.abspath('src'))

mock_hivemind = mock.MagicMock()
mock_petals_utils_misc = mock.MagicMock()
mock_petals_utils_misc.DUMMY = torch.empty(0)

mock_tensor_parallel = mock.MagicMock()

with mock.patch.dict('sys.modules', {
    'hivemind': mock_hivemind,
    'hivemind.get_logger': mock_hivemind.get_logger,
    'hivemind.moe': mock_hivemind.moe,
    'hivemind.moe.client': mock_hivemind.moe.client,
    'hivemind.moe.client.remote_expert_worker': mock_hivemind.moe.client.remote_expert_worker,
    'hivemind.moe.expert_uid': mock_hivemind.moe.expert_uid,
    'hivemind.moe.server': mock_hivemind.moe.server,
    'hivemind.moe.server.connection_handler': mock_hivemind.moe.server.connection_handler,
    'hivemind.moe.server.module_backend': mock_hivemind.moe.server.module_backend,
    'hivemind.p2p': mock_hivemind.p2p,
    'hivemind.p2p.p2p_daemon_bindings': mock_hivemind.p2p.p2p_daemon_bindings,
    'hivemind.p2p.p2p_daemon_bindings.control': mock_hivemind.p2p.p2p_daemon_bindings.control,
    'hivemind.p2p.p2p_daemon': mock_hivemind.p2p.p2p_daemon,
    'hivemind.compression': mock_hivemind.compression,
    'hivemind.compression.quantization': mock_hivemind.compression.quantization,
    'hivemind.compression.serialization': mock_hivemind.compression.serialization,
    'hivemind.utils': mock_hivemind.utils,
    'hivemind.utils.logging': mock_hivemind.utils.logging,
    'hivemind.utils.tensor_deserializer': mock_hivemind.utils.tensor_deserializer,
    'hivemind.utils.tensor_descr': mock_hivemind.utils.tensor_descr,
    'hivemind.utils.streaming': mock_hivemind.utils.streaming,
    'hivemind.utils.asyncio': mock_hivemind.utils.asyncio,
    'hivemind.utils.mpfuture': mock_hivemind.utils.mpfuture,
    'hivemind.utils.nested': mock_hivemind.utils.nested,
    'hivemind.proto': mock_hivemind.proto,
    'hivemind.proto.runtime_pb2': mock_hivemind.proto.runtime_pb2,
    'hivemind.dht': mock_hivemind.dht,
    'hivemind.dht.node': mock_hivemind.dht.node,
    'hivemind.utils.networking': mock_hivemind.utils.networking,
    'petals.utils.misc': mock_petals_utils_misc,
    'tensor_parallel': mock_tensor_parallel,
    'tensor_parallel.tensor_parallel': mock_tensor_parallel.tensor_parallel,
    'tensor_parallel.slicing_configs': mock_tensor_parallel.slicing_configs,
    'speedtest': mock.MagicMock(),
}):
    mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
    mock_hivemind.p2p.p2p_daemon_bindings.control.MAX_UNARY_PAYLOAD_SIZE = 1000000
    mock_hivemind.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000

    from transformers import PretrainedConfig
    from petals.client.ptune import PTuneMixin

    class DummyModel(PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = mock.MagicMock()
            self.word_embeddings.weight = torch.zeros(1, dtype=torch.float32)
            self.init_prompts(config)

    def test_ptune_intermediate_prompt_embeddings_shape():
        config = PretrainedConfig()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 10
        config.hidden_size = 64
        config.num_hidden_layers = 12

        model = DummyModel(config)

        assert model.pre_seq_len == 10
        assert model.prompt_embeddings.weight.shape == (10, 64)
        assert model.intermediate_prompt_embeddings.weight.shape == (10, 11 * 64)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 10, 64)
        assert intermediate_prompts.shape == (11, 2, 10, 64)
