import os
import sys
import unittest.mock as mock

# Ensure local petals package takes precedence
sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

def test_ptune_intermediate_shape():
    # Extensive mocking for heavy dependencies localized to the test function
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
        'hivemind.utils': mock.MagicMock(),
        'hivemind.utils.asyncio': mock.MagicMock(),
        'hivemind.utils.mpfuture': mock.MagicMock(),
        'hivemind.utils.streaming': mock.MagicMock(),
        'hivemind.utils.nested': mock.MagicMock(),
        'hivemind.utils.tensor_descr': mock.MagicMock(),
        'hivemind.utils.custom_warnings': mock.MagicMock(),
        'hivemind.utils.logging': mock.MagicMock(),
        'hivemind.compression': mock.MagicMock(),
        'hivemind.compression.serialization': mock.MagicMock(),
        'hivemind.proto': mock.MagicMock(),
        'hivemind.proto.runtime_pb2': mock.MagicMock(),
        'hivemind.dht': mock.MagicMock(),
        'hivemind.dht.crypto': mock.MagicMock(),
        'hivemind.dht.schema': mock.MagicMock(),
        'hivemind.dht.node': mock.MagicMock(),
        'hivemind.dht.routing': mock.MagicMock(),
        'hivemind.dht.validation': mock.MagicMock(),
        'hivemind.optim': mock.MagicMock(),
        'tensor_parallel': mock.MagicMock(),
        'tensor_parallel.slicing_configs': mock.MagicMock(),
        'tensor_parallel.tensor_parallel': mock.MagicMock(),
        'speedtest': mock.MagicMock(),
        'petals.utils.version': mock.MagicMock()
    }

    mocks['hivemind.p2p.p2p_daemon_bindings.control'].DEFAULT_MAX_MSG_SIZE = 1000000
    mocks['hivemind.p2p.p2p_daemon_bindings.control'].MAX_UNARY_PAYLOAD_SIZE = 1000000
    mocks['hivemind.p2p.p2p_daemon'].DEFAULT_MAX_MSG_SIZE = 1000000

    with mock.patch.dict('sys.modules', mocks):
        import torch
        from transformers import PretrainedConfig
        from petals.client.ptune import PTuneMixin

        class DummyModel(torch.nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=10
        )

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 16])
        assert intermediate_prompts.shape == torch.Size([9, 2, 5, 16])
