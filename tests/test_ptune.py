import sys
import os
import torch
import unittest.mock as mock
import pytest

mock_hivemind = mock.MagicMock()
mocks = {
    'hivemind': mock_hivemind,
    'hivemind.moe': mock_hivemind,
    'hivemind.moe.client': mock_hivemind,
    'hivemind.moe.client.remote_expert_worker': mock_hivemind,
    'hivemind.moe.server': mock_hivemind,
    'hivemind.moe.server.module_backend': mock_hivemind,
    'hivemind.moe.server.connection_handler': mock_hivemind,
    'hivemind.moe.expert_uid': mock_hivemind,
    'hivemind.p2p': mock_hivemind,
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
    'tensor_parallel': mock_hivemind,
    'tensor_parallel.slicing_configs': mock_hivemind,
    'tensor_parallel.tensor_parallel': mock_hivemind,
}

def test_ptune_intermediate_prompt_shape():
    with mock.patch.dict('sys.modules', mocks):
        from transformers import PretrainedConfig
        os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'
        sys.path.insert(0, os.path.abspath('src'))

        with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
            from petals.client.ptune import PTuneMixin

            class DummyModel(PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = type('DummyEmbed', (), {'weight': torch.randn((10, config.hidden_size), dtype=torch.float32)})
                    self.init_prompts(config)

            config = PretrainedConfig(
                tuning_mode="deep_ptune",
                pre_seq_len=5,
                hidden_size=16,
                num_hidden_layers=3
            )

            model = DummyModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert prompts.shape == torch.Size([2, 5, 16])
            assert intermediate_prompts.shape == torch.Size([2, 2, 5, 16]) # We expect this to fail before the fix, as it is [3, ...] instead of [2, ...]
