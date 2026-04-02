import sys
from unittest.mock import MagicMock

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

mock_modules = {
    'hivemind': MockPackage(),
    'hivemind.p2p': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings.datastructures': MockPackage(),
    'hivemind.moe': MockPackage(),
    'hivemind.moe.server': MockPackage(),
    'hivemind.moe.server.connection_handler': MockPackage(),
    'hivemind.moe.server.module_backend': MockPackage(),
    'hivemind.moe.expert_uid': MockPackage(),
    'hivemind.moe.client': MockPackage(),
    'hivemind.moe.client.remote_expert_worker': MockPackage(),
    'hivemind.utils': MockPackage(),
    'hivemind.utils.nested': MockPackage(),
    'hivemind.utils.tensor_descr': MockPackage(),
    'hivemind.utils.asyncio': MockPackage(),
    'hivemind.utils.logging': MockPackage(),
    'hivemind.utils.streaming': MockPackage(),
    'hivemind.utils.mpfuture': MockPackage(),
    'hivemind.compression': MockPackage(),
    'hivemind.compression.serialization': MockPackage(),
    'hivemind.dht': MockPackage(),
    'hivemind.dht.routing': MockPackage(),
    'hivemind.dht.node': MockPackage(),
    'hivemind.p2p.p2p_daemon': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings.control': MockPackage(),
    'hivemind.proto': MockPackage(),
    'hivemind.proto.runtime_pb2': MockPackage(),
    'tensor_parallel': MockPackage(),
    'tensor_parallel.slicing_configs': MockPackage(),
    'tensor_parallel.tensor_parallel': MockPackage(),
}

sys.modules.update(mock_modules)

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=5,
    )
    model = DummyModel(config)
    batch_size = 2

    # Call get_prompt
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check prompts shape
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # Check intermediate_prompts shape
    assert intermediate_prompts.shape == (
        config.num_hidden_layers,
        batch_size,
        config.pre_seq_len,
        config.hidden_size,
    )

    # Verify first layer is zero-padded
    assert torch.all(intermediate_prompts[0] == 0)

    # Verify other layers are not necessarily zero
    # (Although they are initialized randomly, so they are practically non-zero)
    # Check that they match what is expected from the embedding output
    prefix_tokens = torch.arange(config.pre_seq_len).long().unsqueeze(0).expand(batch_size, -1)
    expected_intermediate = model.intermediate_prompt_embeddings(prefix_tokens)
    expected_intermediate = expected_intermediate.view(
        batch_size,
        config.pre_seq_len,
        config.num_hidden_layers - 1,
        config.hidden_size,
    )
    expected_intermediate = expected_intermediate.permute([2, 0, 1, 3])

    assert torch.allclose(intermediate_prompts[1:], expected_intermediate.to(prompts.dtype))

if __name__ == "__main__":
    test_deep_ptune()
    print("All tests passed.")
