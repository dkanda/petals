import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.utils.misc import DUMMY

def setup_module():
    class MockPackage(MagicMock):
        __path__ = []
        __spec__ = None

    import unittest.mock
    mock_modules = {
        "hivemind": MockPackage(),
        "hivemind.moe": MockPackage(),
        "hivemind.moe.client": MockPackage(),
        "hivemind.moe.client.remote_expert_worker": MockPackage(),
        "hivemind.p2p": MockPackage(),
        "hivemind.p2p.p2p_daemon": MockPackage(),
        "hivemind.utils": MockPackage(),
        "hivemind.utils.tensor_deserializer": MockPackage(),
        "hivemind.utils.tensor_descr": MockPackage(),
        "hivemind.proto": MockPackage(),
        "hivemind.dht": MockPackage(),
        "hivemind.dht.node": MockPackage(),
        "hivemind.dht.routing": MockPackage(),
        "hivemind.utils.nested": MockPackage(),
        "hivemind.utils.logging": MockPackage(),
        "hivemind.p2p.p2p_daemon_bindings": MockPackage(),
        "hivemind.p2p.p2p_daemon_bindings.datastructures": MockPackage(),
        "hivemind.p2p.p2p_daemon_bindings.utils": MockPackage(),
        "hivemind.p2p.p2p_daemon_bindings.p2pclient": MockPackage(),
        "hivemind.moe.expert_uid": MockPackage(),
        "hivemind.moe.server": MockPackage(),
        "hivemind.moe.server.connection_handler": MockPackage(),
        "hivemind.moe.server.module_backend": MockPackage(),
        "hivemind.moe.server.task_pool": MockPackage(),
        "hivemind.utils.mpfuture": MockPackage(),
        "hivemind.utils.asyncio": MockPackage(),
        "hivemind.utils.streaming": MockPackage(),
        "hivemind.p2p.p2p_daemon_bindings.control": MockPackage(),
        "hivemind.compression": MockPackage(),
        "hivemind.compression.serialization": MockPackage(),
        "hivemind.compression.quantization": MockPackage(),
        "tensor_parallel": MockPackage(),
        "tensor_parallel.slicing_configs": MockPackage(),
        "tensor_parallel.tensor_parallel": MockPackage(),
        "speedtest": MockPackage(),
    }

    global mock_context
    mock_context = unittest.mock.patch.dict("sys.modules", mock_modules)
    mock_context.start()


def teardown_module():
    global mock_context
    mock_context.stop()


class DummyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    from petals.client.ptune import PTuneMixin

    class PTunedDummyModel(DummyModel, PTuneMixin):
        pass

    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=3,
        tuning_mode="ptune",
        pre_seq_len=5,
    )
    model = PTunedDummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 64)
    assert intermediate_prompts is DUMMY or torch.allclose(intermediate_prompts, DUMMY)


def test_deep_ptune_shapes():
    from petals.client.ptune import PTuneMixin

    class PTunedDummyModel(DummyModel, PTuneMixin):
        pass

    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=3,
        tuning_mode="deep_ptune",
        pre_seq_len=5,
    )
    model = PTunedDummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 64)
    assert intermediate_prompts.shape == (3, 2, 5, 64)

    # Check that the first layer's intermediate prompt is zero-padded
    assert torch.all(intermediate_prompts[0] == 0)

    # Check that other layers are not entirely zeros (they are initialized with some values)
    assert not torch.all(intermediate_prompts[1] == 0)
    assert not torch.all(intermediate_prompts[2] == 0)
