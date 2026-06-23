import sys
from unittest import mock
import os

# Put source path first so isolated test imports local files correctly
sys.path.insert(0, os.path.abspath('src'))

# Make pytest ignore uninstalled optional deps if we test locally
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

import torch
import torch.nn as nn
from transformers import PretrainedConfig

# Mock heavy deep dependencies for the local test
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
    'tensor_parallel': mock.MagicMock(),
    'tensor_parallel.slicing_configs': mock.MagicMock(),
    'tensor_parallel.tensor_parallel': mock.MagicMock()
}

def test_ptune_intermediate_prompt_embeddings_shape():
    """Verify that deep ptune intermediate prompts correctly generate shape (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)."""
    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin

        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                # Ensure word_embeddings exists so dtype/device can be inferred
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        # Deep Ptune configuration
        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=3
        )

        model = DummyModel(config)
        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

        # Intermediate prompts are sent for layers > 0, so there are (num_hidden_layers - 1) of them
        expected_shape = torch.Size([config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size])
        assert intermediate_prompts.shape == expected_shape, f"Expected shape {expected_shape}, got {intermediate_prompts.shape}"
