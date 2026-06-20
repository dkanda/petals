import sys
import os
sys.path.insert(0, os.path.abspath('src'))
from unittest import mock

mocks = {
    'hivemind': mock.MagicMock(),
    'hivemind.compression': mock.MagicMock(),
    'hivemind.moe': mock.MagicMock(),
    'hivemind.moe.client': mock.MagicMock(),
    'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
    'hivemind.p2p': mock.MagicMock(),
    'hivemind.dht': mock.MagicMock(),
    'hivemind.moe.server': mock.MagicMock(),
    'hivemind.moe.server.module_backend': mock.MagicMock(),
    'hivemind.moe.server.connection_handler': mock.MagicMock(),
    'hivemind.moe.expert_uid': mock.MagicMock(),
    'hivemind.p2p.p2p_daemon_bindings': mock.MagicMock(),
    'hivemind.p2p.p2p_daemon_bindings.control': mock.MagicMock(),
    'hivemind.p2p.p2p_daemon': mock.MagicMock(),
    'hivemind.dht.node': mock.MagicMock(),
    'hivemind.proto': mock.MagicMock(),
    'hivemind.proto.runtime_pb2': mock.MagicMock(),
    'hivemind.utils': mock.MagicMock(),
    'hivemind.utils.logging': mock.MagicMock(),
    'hivemind.utils.asyncio': mock.MagicMock(),
    'hivemind.utils.mpfuture': mock.MagicMock(),
    'hivemind.utils.streaming': mock.MagicMock(),
    'hivemind.utils.nested': mock.MagicMock(),
    'hivemind.utils.tensor_descr': mock.MagicMock(),
    'hivemind.compression.serialization': mock.MagicMock(),
    'tensor_parallel.slicing_configs': mock.MagicMock(),
    'tensor_parallel.tensor_parallel': mock.MagicMock(),
    'tensor_parallel': mock.MagicMock()
}

mock_misc = mock.MagicMock()
import torch
mock_misc.DUMMY = torch.empty(0)
mocks['petals.utils.misc'] = mock_misc

with mock.patch.dict('sys.modules', mocks):
    import torch.nn as nn
    from transformers import PretrainedConfig
    from petals.client.ptune import PTuneMixin

    class DummyModel(PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = nn.Embedding(10, config.hidden_size)
            self.init_prompts(config)

def test_ptune_mixin_deep_ptune():
    config = PretrainedConfig(
        hidden_size=8,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=5
    )

    with mock.patch.dict('sys.modules', mocks):
        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        # Test shapes
        assert prompts.shape == torch.Size([2, 5, 8])
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 8])

def test_ptune_mixin_ptune():
    config = PretrainedConfig(
        hidden_size=8,
        num_hidden_layers=4,
        tuning_mode="ptune",
        pre_seq_len=5
    )

    with mock.patch.dict('sys.modules', mocks):
        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        # Test shapes
        assert prompts.shape == torch.Size([2, 5, 8])
        assert intermediate_prompts.shape == torch.Size([0])
