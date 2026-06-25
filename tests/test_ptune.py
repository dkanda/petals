import os
import sys
sys.path.insert(0, os.path.abspath('src'))
from unittest import mock
import torch

def test_ptune_intermediate_shape():
    mock_hivemind = mock.MagicMock()
    mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
    mock_hivemind.p2p.p2p_daemon_bindings.control.MAX_UNARY_PAYLOAD_SIZE = 1000000
    mock_hivemind.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000

    mocks = {
        'hivemind': mock_hivemind,
        'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
        'hivemind.moe.server.module_backend': mock.MagicMock(),
        'hivemind.moe.server.connection_handler': mock.MagicMock(),
        'hivemind.moe.expert_uid': mock.MagicMock(),
        'hivemind.p2p': mock_hivemind.p2p,
        'hivemind.p2p.p2p_daemon_bindings': mock_hivemind.p2p.p2p_daemon_bindings,
        'hivemind.p2p.p2p_daemon_bindings.control': mock_hivemind.p2p.p2p_daemon_bindings.control,
        'hivemind.p2p.p2p_daemon': mock_hivemind.p2p.p2p_daemon,
        'hivemind.dht.node': mock.MagicMock(),
        'hivemind.dht': mock.MagicMock(),
        'hivemind.proto': mock.MagicMock(),
        'hivemind.proto.runtime_pb2': mock.MagicMock(),
        'hivemind.utils': mock.MagicMock(),
        'hivemind.utils.asyncio': mock.MagicMock(),
        'hivemind.utils.logging': mock.MagicMock(),
        'hivemind.utils.mpfuture': mock.MagicMock(),
        'hivemind.utils.streaming': mock.MagicMock(),
        'hivemind.utils.nested': mock.MagicMock(),
        'hivemind.utils.tensor_descr': mock.MagicMock(),
        'hivemind.compression': mock.MagicMock(),
        'hivemind.compression.serialization': mock.MagicMock(),
        'hivemind.compression.quantization': mock.MagicMock(),
        'tensor_parallel': mock.MagicMock(),
        'tensor_parallel.slicing_configs': mock.MagicMock(),
        'tensor_parallel.tensor_parallel': mock.MagicMock(),
    }

    with mock.patch.dict('sys.modules', mocks):
        from transformers import PretrainedConfig
        from petals.client.ptune import PTuneMixin
        import petals

        class DummyModel(PTuneMixin, torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=3
        )
        model = DummyModel(config)

        # Test shape of initialized embeddings
        assert model.intermediate_prompt_embeddings.weight.shape == torch.Size([5, 32]), f"Expected [5, 32], got {model.intermediate_prompt_embeddings.weight.shape}"

        # Test generated prompt shapes
        prompts, intermediate = model.get_prompt(batch_size=2)
        assert prompts.shape == torch.Size([2, 5, 16]), f"Expected [2, 5, 16], got {prompts.shape}"
        assert intermediate.shape == torch.Size([2, 2, 5, 16]), f"Expected [2, 2, 5, 16], got {intermediate.shape}"

if __name__ == '__main__':
    test_ptune_intermediate_shape()
    print("All tests passed.")
