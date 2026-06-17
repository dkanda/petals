import sys
import os

sys.path.insert(0, os.path.abspath('src'))
from unittest import mock
import torch

def test_ptune_intermediate_prompt_shape():
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
        'hivemind.p2p.p2p_daemon': mock.MagicMock(),
        'hivemind.p2p.p2p_daemon_bindings': mock.MagicMock(),
        'hivemind.p2p.p2p_daemon_bindings.control': mock.MagicMock(),
        'hivemind.utils': mock.MagicMock(),
        'hivemind.utils.asyncio': mock.MagicMock(),
        'hivemind.utils.streaming': mock.MagicMock(),
        'hivemind.utils.nested': mock.MagicMock(),
        'hivemind.utils.tensor_descr': mock.MagicMock(),
        'hivemind.compression': mock.MagicMock(),
        'hivemind.compression.base': mock.MagicMock(),
        'hivemind.compression.serialization': mock.MagicMock(),
        'hivemind.proto': mock.MagicMock(),
        'hivemind.proto.runtime_pb2': mock.MagicMock(),
        'hivemind.dht': mock.MagicMock(),
        'hivemind.dht.node': mock.MagicMock(),
        'hivemind.utils.logging': mock.MagicMock(),
        'hivemind.utils.mpfuture': mock.MagicMock(),
        'hivemind.utils.serializer': mock.MagicMock(),
        'tensor_parallel': mock.MagicMock(),
        'tensor_parallel.tensor_parallel': mock.MagicMock(),
        'tensor_parallel.slicing_configs': mock.MagicMock(),
    }

    mocks['hivemind.p2p.p2p_daemon'].DEFAULT_MAX_MSG_SIZE = 1000000
    mocks['hivemind.p2p.p2p_daemon_bindings.control'].MAX_UNARY_PAYLOAD_SIZE = 1000000

    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin

        class DummyConfig:
            tuning_mode = 'deep_ptune'
            pre_seq_len = 10
            hidden_size = 32
            num_hidden_layers = 12

        class DummyModel(PTuneMixin):
            def __init__(self):
                self.config = DummyConfig()
                self.init_prompts(self.config)

                # mock word embeddings
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight.device = 'cpu'
                self.word_embeddings.weight.dtype = torch.float32

        model = DummyModel()

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

        # Expected shape: (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
        expected_shape = (11, batch_size, 10, 32)
        assert intermediate_prompts.shape == expected_shape, f"Expected shape {expected_shape}, but got {intermediate_prompts.shape}"

if __name__ == '__main__':
    test_ptune_intermediate_prompt_shape()
    print("Test passed successfully.")
