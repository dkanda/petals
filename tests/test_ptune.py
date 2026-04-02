import unittest.mock
import sys

# Need to mock hivemind and tensor_parallel due to dependencies issues
class MockPackage:
    def __init__(self):
        self.__path__ = []
        self.__spec__ = None

    def __getattr__(self, item):
        return unittest.mock.MagicMock()

mock_modules = {
    'hivemind': MockPackage(),
    'hivemind.p2p': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings.control': MockPackage(),
    'hivemind.p2p.p2p_daemon': MockPackage(),
    'hivemind.moe': MockPackage(),
    'hivemind.moe.client': MockPackage(),
    'hivemind.moe.client.remote_expert_worker': MockPackage(),
    'hivemind.moe.server': MockPackage(),
    'hivemind.moe.server.connection_handler': MockPackage(),
    'hivemind.moe.server.module_backend': MockPackage(),
    'hivemind.moe.expert_uid': MockPackage(),
    'hivemind.utils': MockPackage(),
    'hivemind.utils.nested': MockPackage(),
    'hivemind.utils.tensor_descr': MockPackage(),
    'hivemind.utils.streaming': MockPackage(),
    'hivemind.utils.logging': MockPackage(),
    'hivemind.utils.asyncio': MockPackage(),
    'hivemind.utils.mpfuture': MockPackage(),
    'hivemind.compression': MockPackage(),
    'hivemind.compression.serialization': MockPackage(),
    'hivemind.dht': MockPackage(),
    'hivemind.dht.routing': MockPackage(),
    'hivemind.dht.node': MockPackage(),
    'hivemind.proto': MockPackage(),
    'hivemind.proto.runtime_pb2': MockPackage(),
    'tensor_parallel': MockPackage(),
    'tensor_parallel.slicing_configs': MockPackage(),
    'tensor_parallel.tensor_parallel': MockPackage(),
}

# Start the mock immediately so it's active during pytest collection
patcher = unittest.mock.patch.dict('sys.modules', mock_modules)
patcher.start()

def teardown_module():
    patcher.stop()


import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

# We import the mixin directly as petals initialization might trigger dependencies we haven't mocked perfectly
from petals.client.ptune import PTuneMixin, PTuneConfig


class DummyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 16
        self.num_hidden_layers = 4
        self.pre_seq_len = 5
        self.tuning_mode = "deep_ptune"


class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)


def test_deep_ptune_shapes_and_padding():
    config = DummyConfig()
    model = DummyModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # 1. Check prompt shapes
    # prompts: [batch_size, pre_seq_len, hidden_size]
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # 2. Check intermediate_prompts shapes
    # intermediate_prompts: [num_hidden_layers, batch_size, pre_seq_len, hidden_size]
    assert intermediate_prompts.shape == (
        config.num_hidden_layers,
        batch_size,
        config.pre_seq_len,
        config.hidden_size,
    )

    # 3. Check that the first layer's intermediate prompt is fully zeroed out (padding)
    first_layer_prompts = intermediate_prompts[0]
    assert torch.all(first_layer_prompts == 0), "The first layer of intermediate_prompts should be all zeros."

    # 4. Check that the subsequent layers are not all zeros (they are initialized embeddings)
    subsequent_layers = intermediate_prompts[1:]
    assert not torch.all(subsequent_layers == 0), "Subsequent layers should have initialized embeddings."

    # 5. Check dtypes
    assert intermediate_prompts.dtype == model.word_embeddings.weight.dtype
