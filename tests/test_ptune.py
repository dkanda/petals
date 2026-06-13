import os
import sys
from unittest import mock

import torch
import torch.nn as nn
from transformers import PretrainedConfig

# Add src to the path to resolve local modules
sys.path.insert(0, os.path.abspath("src"))

def test_ptune_mixin_intermediate_embeddings():
    # Mock external dependencies that fail to install in pytest environment
    hivemind_mock = mock.MagicMock()
    tensor_parallel_mock = mock.MagicMock()
    transformers_mock = mock.MagicMock()
    transformers_mock.__version__ = "4.43.1"

    os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

    mocks = {
        "hivemind": hivemind_mock,
        "hivemind.compression": hivemind_mock,
        "hivemind.compression.serialization": hivemind_mock,
        "hivemind.moe": hivemind_mock,
        "hivemind.moe.client": hivemind_mock,
        "hivemind.moe.client.remote_expert_worker": hivemind_mock,
        "hivemind.moe.expert_uid": hivemind_mock,
        "hivemind.moe.server": hivemind_mock,
        "hivemind.moe.server.connection_handler": hivemind_mock,
        "hivemind.moe.server.module_backend": hivemind_mock,
        "hivemind.p2p": hivemind_mock,
        "hivemind.p2p.p2p_daemon": hivemind_mock,
        "hivemind.p2p.p2p_daemon_bindings": hivemind_mock,
        "hivemind.p2p.p2p_daemon_bindings.control": hivemind_mock,
        "hivemind.proto": hivemind_mock,
        "hivemind.utils": hivemind_mock,
        "hivemind.utils.mpfuture": hivemind_mock,
        "hivemind.utils.tensor_descr": hivemind_mock,
        "hivemind.utils.crypto": hivemind_mock,
        "hivemind.utils.networking": hivemind_mock,
        "hivemind.utils.logging": hivemind_mock,
        "hivemind.utils.asyncio": hivemind_mock,
        "hivemind.utils.streaming": hivemind_mock,
        "hivemind.utils.nested": hivemind_mock,
        "hivemind.dht": hivemind_mock,
        "hivemind.dht.node": hivemind_mock,
        "hivemind.dht.crypto": hivemind_mock,
        "hivemind.dht.routing": hivemind_mock,
        "tensor_parallel": tensor_parallel_mock,
        "tensor_parallel.tensor_parallel": tensor_parallel_mock,
        "tensor_parallel.slicing_configs": tensor_parallel_mock,
        "tensor_parallel.cross_device_ops": tensor_parallel_mock,
        "speedtest": mock.MagicMock(),
        "petals.utils.misc": mock.MagicMock(DUMMY=torch.empty(0)),
    }

    with mock.patch.dict("sys.modules", mocks), mock.patch("transformers.utils.import_utils.is_torch_fx_available", return_value=False, create=True):
        from petals.client.ptune import PTuneMixin

        class MockConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.tuning_mode = "deep_ptune"
                self.pre_seq_len = 10
                self.hidden_size = 32
                self.num_hidden_layers = 12

        class TestModel(PTuneMixin):
            def __init__(self):
                self.config = MockConfig()
                self.word_embeddings = nn.Embedding(100, 32)

        model = TestModel()
        model.init_prompts(model.config)

        # Verify prompt embeddings shape
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        # prompts should be of shape (batch_size, pre_seq_len, hidden_size)
        assert prompts.shape == (2, 10, 32)

        # intermediate_prompts should be of shape (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
        assert intermediate_prompts.shape == (11, 2, 10, 32)
