import torch
import sys
import os
from unittest import mock

# Ensure src is on path for isolated run if needed
sys.path.insert(0, os.path.abspath('src'))

# Mock missing heavy dependencies and package parts for isolated testing
mock_hivemind = mock.MagicMock()
mock_hivemind.utils = mock.MagicMock()
mock_hivemind.utils.streaming = mock.MagicMock()
mock_hivemind.utils.nested = mock.MagicMock()
mock_hivemind.utils.asyncio = mock.MagicMock()
mock_hivemind.utils.tensor_descr = mock.MagicMock()
mock_hivemind.utils.mpfuture = mock.MagicMock()
mock_hivemind.p2p = mock.MagicMock()
mock_hivemind.p2p.p2p_daemon_bindings = mock.MagicMock()
mock_hivemind.p2p.p2p_daemon_bindings.control = mock.MagicMock()
mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
mock_hivemind.p2p.p2p_daemon_bindings.control.MAX_UNARY_PAYLOAD_SIZE = 1000000
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
mock_hivemind.proto = mock.MagicMock()
mock_hivemind.proto.runtime_pb2 = mock.MagicMock()
mock_hivemind.dht = mock.MagicMock()
mock_hivemind.dht.node = mock.MagicMock()
mock_hivemind.utils.logging = mock.MagicMock()
mock_hivemind.utils.logging.get_logger = mock.MagicMock(return_value=mock.MagicMock())

mock_tensor_parallel = mock.MagicMock()
mock_tensor_parallel.tensor_parallel = mock.MagicMock()

mock_modules = {
    'hivemind': mock_hivemind,
    'hivemind.utils': mock_hivemind.utils,
    'hivemind.utils.streaming': mock_hivemind.utils.streaming,
    'hivemind.utils.nested': mock_hivemind.utils.nested,
    'hivemind.utils.asyncio': mock_hivemind.utils.asyncio,
    'hivemind.utils.tensor_descr': mock_hivemind.utils.tensor_descr,
    'hivemind.utils.mpfuture': mock_hivemind.utils.mpfuture,
    'hivemind.utils.logging': mock_hivemind.utils.logging,
    'hivemind.p2p': mock_hivemind.p2p,
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
    'hivemind.proto': mock_hivemind.proto,
    'hivemind.proto.runtime_pb2': mock_hivemind.proto.runtime_pb2,
    'hivemind.dht': mock_hivemind.dht,
    'hivemind.dht.node': mock_hivemind.dht.node,
    'tensor_parallel': mock_tensor_parallel,
    'tensor_parallel.slicing_configs': mock_tensor_parallel.slicing_configs,
    'tensor_parallel.tensor_parallel': mock_tensor_parallel.tensor_parallel,
}

def test_ptune_intermediate_prompt_shape():
    """
    Test that the intermediate_prompt_embeddings in deep_ptune mode
    correctly allocates and outputs shape scaled by (num_hidden_layers - 1).
    """
    with mock.patch.dict(sys.modules, mock_modules):
        from petals.client.ptune import PTuneMixin
        from transformers import PretrainedConfig

        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight = torch.zeros(1, 1, dtype=torch.float32)
                self.init_prompts(config)

        # Configure model
        config = PretrainedConfig()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 10
        config.hidden_size = 64
        config.num_hidden_layers = 12
        batch_size = 2

        model = DummyModel(config)

        # Test shape of the embedding parameter itself
        expected_embedding_dim = (config.num_hidden_layers - 1) * config.hidden_size
        assert model.intermediate_prompt_embeddings.weight.shape == (config.pre_seq_len, expected_embedding_dim), \
            f"Expected embedding shape {(config.pre_seq_len, expected_embedding_dim)}, got {model.intermediate_prompt_embeddings.weight.shape}"

        # Test shape of the returned outputs
        prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

        expected_prompts_shape = (batch_size, config.pre_seq_len, config.hidden_size)
        assert prompts.shape == expected_prompts_shape, \
            f"Expected prompts shape {expected_prompts_shape}, got {prompts.shape}"

        expected_intermediate_shape = (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
        assert intermediate_prompts.shape == expected_intermediate_shape, \
            f"Expected intermediate_prompts shape {expected_intermediate_shape}, got {intermediate_prompts.shape}"

if __name__ == "__main__":
    test_ptune_intermediate_prompt_shape()
    print("All tests passed.")
