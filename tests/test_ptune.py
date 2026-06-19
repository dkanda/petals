import sys
import os
from unittest import mock
import torch

def test_ptune_deep_ptune_dimensions():
    # Mock required deep dependencies to bypass ModuleNotFoundError
    mocks = {
        'hivemind': mock.MagicMock(),
        'hivemind.moe': mock.MagicMock(),
        'hivemind.moe.server': mock.MagicMock(),
        'hivemind.moe.server.module_backend': mock.MagicMock(),
        'hivemind.moe.server.connection_handler': mock.MagicMock(),
        'hivemind.moe.expert_uid': mock.MagicMock(),
        'hivemind.moe.client': mock.MagicMock(),
        'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
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
        'hivemind.utils.mpfuture': mock.MagicMock(),
        'hivemind.utils.streaming': mock.MagicMock(),
        'hivemind.utils.nested': mock.MagicMock(),
        'hivemind.utils.tensor_descr': mock.MagicMock(),
        'hivemind.utils.logging': mock.MagicMock(),
        'hivemind.compression': mock.MagicMock(),
        'hivemind.compression.serialization': mock.MagicMock(),
        'tensor_parallel': mock.MagicMock(),
        'tensor_parallel.slicing_configs': mock.MagicMock(),
        'tensor_parallel.tensor_parallel': mock.MagicMock(),
        'petals.utils.misc': mock.MagicMock()
    }

    with mock.patch.dict('sys.modules', mocks):
        mocks['petals.utils.misc'].DUMMY = torch.empty(0)
        mocks['hivemind.p2p.p2p_daemon_bindings.control'].DEFAULT_MAX_MSG_SIZE = 1000000
        mocks['hivemind.p2p.p2p_daemon_bindings.control'].MAX_UNARY_PAYLOAD_SIZE = 1000000
        mocks['hivemind.p2p.p2p_daemon'].DEFAULT_MAX_MSG_SIZE = 1000000

        from transformers import PretrainedConfig
        from petals.client.ptune import PTuneMixin

        class MockConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.tuning_mode = kwargs.get("tuning_mode", None)
                self.pre_seq_len = kwargs.get("pre_seq_len", 0)
                self.hidden_size = kwargs.get("hidden_size", 16)
                self.num_hidden_layers = kwargs.get("num_hidden_layers", 4)

        class MockModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                class MockWordEmbeddings:
                    weight = mock.MagicMock()
                    weight.device = "cpu"
                    weight.dtype = torch.float32
                self.word_embeddings = MockWordEmbeddings()
                self.init_prompts(config)

        config = MockConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=4)
        model = MockModel(config)

        # Check that intermediate embeddings have size for (num_hidden_layers - 1)
        expected_embedding_size = (config.num_hidden_layers - 1) * config.hidden_size
        assert model.intermediate_prompt_embeddings.weight.shape == (config.pre_seq_len, expected_embedding_size)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

        assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
        expected_intermediate_shape = (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
        assert intermediate_prompts.shape == expected_intermediate_shape


def test_ptune_normal_ptune_dimensions():
    mocks = {
        'hivemind': mock.MagicMock(),
        'hivemind.moe': mock.MagicMock(),
        'hivemind.moe.server': mock.MagicMock(),
        'hivemind.moe.server.module_backend': mock.MagicMock(),
        'hivemind.moe.server.connection_handler': mock.MagicMock(),
        'hivemind.moe.expert_uid': mock.MagicMock(),
        'hivemind.moe.client': mock.MagicMock(),
        'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
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
        'hivemind.utils.mpfuture': mock.MagicMock(),
        'hivemind.utils.streaming': mock.MagicMock(),
        'hivemind.utils.nested': mock.MagicMock(),
        'hivemind.utils.tensor_descr': mock.MagicMock(),
        'hivemind.utils.logging': mock.MagicMock(),
        'hivemind.compression': mock.MagicMock(),
        'hivemind.compression.serialization': mock.MagicMock(),
        'tensor_parallel': mock.MagicMock(),
        'tensor_parallel.slicing_configs': mock.MagicMock(),
        'tensor_parallel.tensor_parallel': mock.MagicMock(),
        'petals.utils.misc': mock.MagicMock()
    }

    with mock.patch.dict('sys.modules', mocks):
        mocks['petals.utils.misc'].DUMMY = torch.empty(0)
        mocks['hivemind.p2p.p2p_daemon_bindings.control'].DEFAULT_MAX_MSG_SIZE = 1000000
        mocks['hivemind.p2p.p2p_daemon_bindings.control'].MAX_UNARY_PAYLOAD_SIZE = 1000000
        mocks['hivemind.p2p.p2p_daemon'].DEFAULT_MAX_MSG_SIZE = 1000000

        from transformers import PretrainedConfig
        from petals.client.ptune import PTuneMixin

        class MockConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.tuning_mode = kwargs.get("tuning_mode", None)
                self.pre_seq_len = kwargs.get("pre_seq_len", 0)
                self.hidden_size = kwargs.get("hidden_size", 16)
                self.num_hidden_layers = kwargs.get("num_hidden_layers", 4)

        class MockModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                class MockWordEmbeddings:
                    weight = mock.MagicMock()
                    weight.device = "cpu"
                    weight.dtype = torch.float32
                self.word_embeddings = MockWordEmbeddings()
                self.init_prompts(config)

        config = MockConfig(tuning_mode="ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=4)
        model = MockModel(config)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

        assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
        assert intermediate_prompts.numel() == 0
