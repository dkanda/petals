import os
import sys
from unittest import mock

import torch
import torch.nn as nn
from transformers import PretrainedConfig

# Mock modules locally for testing environments without these deep dependencies
mocks = {
    'hivemind': mock.MagicMock(),
    'tensor_parallel': mock.MagicMock(),
    'tensor_parallel.tensor_parallel': mock.MagicMock(),
    'tensor_parallel.slicing_configs': mock.MagicMock(),
    'hivemind.moe': mock.MagicMock(),
    'hivemind.moe.client': mock.MagicMock(),
    'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
    'hivemind.moe.expert_uid': mock.MagicMock(),
    'hivemind.utils': mock.MagicMock(),
    'hivemind.utils.asyncio': mock.MagicMock(),
    'hivemind.utils.logging': mock.MagicMock(),
    'hivemind.utils.mpfuture': mock.MagicMock(),
    'hivemind.utils.streaming': mock.MagicMock(),
    'hivemind.utils.nested': mock.MagicMock(),
    'hivemind.p2p': mock.MagicMock(),
    'hivemind.p2p.p2p_daemon_bindings': mock.MagicMock(),
    'hivemind.p2p.p2p_daemon_bindings.control': mock.MagicMock(),
    'hivemind.p2p.p2p_daemon': mock.MagicMock(),
    'hivemind.utils.tensor_descr': mock.MagicMock(),
    'hivemind.moe.server': mock.MagicMock(),
    'hivemind.moe.server.module_backend': mock.MagicMock(),
    'hivemind.moe.server.connection_handler': mock.MagicMock(),
    'hivemind.dht': mock.MagicMock(),
    'hivemind.dht.node': mock.MagicMock(),
    'hivemind.proto': mock.MagicMock(),
    'hivemind.proto.runtime_pb2': mock.MagicMock(),
    'hivemind.compression': mock.MagicMock(),
    'hivemind.compression.serialization': mock.MagicMock(),
}

with mock.patch.dict('sys.modules', mocks):
    with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
        from petals.client.ptune import PTuneMixin

class MyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pre_seq_len = 5
        self.tuning_mode = "deep_ptune"
        self.hidden_size = 10
        self.num_hidden_layers = 4

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_embeddings_shape():
    config = MyConfig()
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    # prefix_tokens: batch_size, pre_seq_len, hidden_size
    assert prompts.shape == torch.Size([2, 5, 10])

    # intermediate_prompts: num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size
    assert intermediate_prompts.shape == torch.Size([3, 2, 5, 10])

    print("PTune intermediate embedding shape tests passed!")

if __name__ == "__main__":
    test_ptune_embeddings_shape()
