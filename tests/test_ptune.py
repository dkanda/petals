import torch
import torch.nn as nn
from transformers import PretrainedConfig
import pytest
import unittest.mock

# Mock dependencies to avoid hivemind import errors in minimal environment
mock_hivemind = unittest.mock.MagicMock()
mock_hivemind.__path__ = []
mock_hivemind.__spec__ = None

mock_tensor_parallel = unittest.mock.MagicMock()
mock_tensor_parallel.__path__ = []
mock_tensor_parallel.__spec__ = None

with unittest.mock.patch.dict('sys.modules', {
    'hivemind': mock_hivemind,
    'hivemind.compression': mock_hivemind,
    'hivemind.compression.serialization': mock_hivemind,
    'hivemind.moe': mock_hivemind,
    'hivemind.moe.client': mock_hivemind,
    'hivemind.moe.client.remote_expert_worker': mock_hivemind,
    'hivemind.moe.expert_uid': mock_hivemind,
    'hivemind.moe.server': mock_hivemind,
    'hivemind.moe.server.connection_handler': mock_hivemind,
    'hivemind.moe.server.module_backend': mock_hivemind,
    'hivemind.p2p': mock_hivemind,
    'hivemind.p2p.p2p_daemon': mock_hivemind,
    'hivemind.p2p.p2p_daemon_bindings': mock_hivemind,
    'hivemind.p2p.p2p_daemon_bindings.control': mock_hivemind,
    'hivemind.p2p.p2p_daemon_bindings.datastructures': mock_hivemind,
    'hivemind.utils': mock_hivemind,
    'hivemind.utils.asyncio': mock_hivemind,
    'hivemind.utils.logging': mock_hivemind,
    'hivemind.utils.mpfuture': mock_hivemind,
    'hivemind.utils.nested': mock_hivemind,
    'hivemind.utils.streaming': mock_hivemind,
    'hivemind.utils.tensor_descr': mock_hivemind,
    'hivemind.dht': mock_hivemind,
    'hivemind.dht.node': mock_hivemind,
    'hivemind.dht.routing': mock_hivemind,
    'hivemind.proto': mock_hivemind,
    'hivemind.proto.runtime_pb2': mock_hivemind,
    'tensor_parallel': mock_tensor_parallel,
    'tensor_parallel.slicing_configs': mock_tensor_parallel,
    'tensor_parallel.tensor_parallel': mock_tensor_parallel,
}):
    from petals.client.ptune import PTuneMixin, PTuneConfig
    from petals.utils.misc import DUMMY

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes_and_padding():
    # Test "deep_ptune" mode
    config_deep = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=5,
        tuning_mode="deep_ptune",
        pre_seq_len=4
    )

    model_deep = MockModel(config_deep)

    # Check intermediate_prompt_embeddings parameter allocation
    # It should be for (num_hidden_layers - 1) * hidden_size
    assert model_deep.intermediate_prompt_embeddings.weight.shape == (
        config_deep.pre_seq_len,
        (config_deep.num_hidden_layers - 1) * config_deep.hidden_size
    )

    batch_size = 2
    prompts, intermediate_prompts = model_deep.get_prompt(batch_size)

    # Check regular prompts shape
    assert prompts.shape == (batch_size, config_deep.pre_seq_len, config_deep.hidden_size)

    # Check intermediate prompts shape (including the padded first layer)
    assert intermediate_prompts.shape == (
        config_deep.num_hidden_layers,
        batch_size,
        config_deep.pre_seq_len,
        config_deep.hidden_size
    )

    # Verify the first layer is explicitly zero-padded
    first_layer_prompts = intermediate_prompts[0]
    assert torch.all(first_layer_prompts == 0)

    # Verify the remaining layers come from the learned embeddings
    # and they should not be strictly zeros natively after initialization
    remaining_layers = intermediate_prompts[1:]
    assert not torch.all(remaining_layers == 0)

def test_ptune_regular_mode():
    # Test "ptune" mode (non-deep)
    config_regular = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=5,
        tuning_mode="ptune",
        pre_seq_len=4
    )

    model_regular = MockModel(config_regular)

    # deep ptune parameters should not be initialized
    assert not hasattr(model_regular, 'intermediate_prompt_embeddings')

    batch_size = 2
    prompts, intermediate_prompts = model_regular.get_prompt(batch_size)

    # Check regular prompts shape
    assert prompts.shape == (batch_size, config_regular.pre_seq_len, config_regular.hidden_size)

    # Verify intermediate prompts is DUMMY
    assert intermediate_prompts is DUMMY or torch.allclose(intermediate_prompts, DUMMY)
