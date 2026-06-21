import sys
import os
import torch
import torch.nn as nn
from unittest import mock

# Standard PyTorch imports executed at the top before sys.modules patching
# to prevent PyTorch from segfaulting.
import transformers

def test_ptune_mixin():
    hivemind_mock = mock.MagicMock()

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.moe': hivemind_mock,
        'hivemind.moe.client': hivemind_mock,
        'hivemind.moe.client.remote_expert_worker': hivemind_mock,
        'hivemind.moe.server': hivemind_mock,
        'hivemind.moe.server.module_backend': hivemind_mock,
        'hivemind.moe.server.connection_handler': hivemind_mock,
        'hivemind.moe.expert_uid': hivemind_mock,
        'hivemind.p2p': hivemind_mock,
        'hivemind.p2p.p2p_daemon_bindings': hivemind_mock,
        'hivemind.p2p.p2p_daemon_bindings.control': hivemind_mock,
        'hivemind.p2p.p2p_daemon': hivemind_mock,
        'hivemind.dht': hivemind_mock,
        'hivemind.dht.node': hivemind_mock,
        'hivemind.proto': hivemind_mock,
        'hivemind.proto.runtime_pb2': hivemind_mock,
        'hivemind.utils': hivemind_mock,
        'hivemind.utils.asyncio': hivemind_mock,
        'hivemind.utils.logging': hivemind_mock,
        'hivemind.utils.mpfuture': hivemind_mock,
        'hivemind.utils.streaming': hivemind_mock,
        'hivemind.utils.nested': hivemind_mock,
        'hivemind.utils.tensor_descr': hivemind_mock,
        'hivemind.compression': hivemind_mock,
        'hivemind.compression.serialization': hivemind_mock,
        'tensor_parallel': hivemind_mock,
        'tensor_parallel.slicing_configs': hivemind_mock,
        'tensor_parallel.tensor_parallel': hivemind_mock,
    }

    hivemind_mock.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
    hivemind_mock.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000

    with mock.patch.dict('sys.modules', mocks):
        import petals.constants

        with mock.patch.object(petals.constants, 'MAX_UNARY_PAYLOAD_SIZE', 1000000):
            from petals.client.ptune import PTuneMixin, PTuneConfig
            from petals.utils.misc import DUMMY

            class MockConfig:
                def __init__(self):
                    self.tuning_mode = "deep_ptune"
                    self.pre_seq_len = 5
                    self.hidden_size = 16
                    self.num_hidden_layers = 10

            class DummyWordEmbeddings:
                def __init__(self):
                    self.weight = torch.empty((0,), dtype=torch.float32)

            class TestModel(PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = DummyWordEmbeddings()
                    self.init_prompts(config)

            config = MockConfig()
            model = TestModel(config)

            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
            assert intermediate_prompts.shape == (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
