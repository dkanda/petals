import sys
import unittest.mock
import pytest

class MockPackage(unittest.mock.MagicMock):
    __path__ = []
    __spec__ = None

def setup_module():
    global _patcher

    mock_modules = {
        'hivemind': MockPackage(),
        'hivemind.moe': MockPackage(),
        'hivemind.moe.expert_uid': MockPackage(),
        'hivemind.moe.client': MockPackage(),
        'hivemind.moe.client.remote_expert_worker': MockPackage(),
        'hivemind.moe.server': MockPackage(),
        'hivemind.moe.server.connection_handler': MockPackage(),
        'hivemind.moe.server.module_backend': MockPackage(),
        'hivemind.p2p': MockPackage(),
        'hivemind.p2p.p2p_daemon': MockPackage(),
        'hivemind.p2p.p2p_daemon_bindings': MockPackage(),
        'hivemind.p2p.p2p_daemon_bindings.control': MockPackage(),
        'hivemind.proto': MockPackage(),
        'hivemind.utils': MockPackage(),
        'hivemind.utils.asyncio': MockPackage(),
        'hivemind.utils.logging': MockPackage(),
        'hivemind.utils.nested': MockPackage(),
        'hivemind.utils.tensor_descr': MockPackage(),
        'hivemind.utils.streaming': MockPackage(),
        'hivemind.utils.mpfuture': MockPackage(),
        'hivemind.compression': MockPackage(),
        'hivemind.compression.base': MockPackage(),
        'hivemind.compression.serialization': MockPackage(),
        'hivemind.dht': MockPackage(),
        'hivemind.dht.node': MockPackage(),
        'hivemind.dht.routing': MockPackage(),
        'hivemind.dht.validation': MockPackage(),
        'tensor_parallel': MockPackage(),
        'tensor_parallel.slicing_configs': MockPackage(),
        'tensor_parallel.tensor_parallel': MockPackage()
    }

    _patcher = unittest.mock.patch.dict(sys.modules, mock_modules)
    _patcher.start()

def teardown_module():
    _patcher.stop()

# We need to defer imports of `petals` to runtime when mock is active
def get_mock_model(config, MockWordEmbeddings):
    from petals.client.ptune import PTuneMixin
    import torch.nn as nn
    import torch

    class MockModel(PTuneMixin, nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = MockWordEmbeddings(torch.float32, torch.device('cpu'))
            self.init_prompts(config)

    return MockModel(config)

def test_ptune_shapes():
    import torch
    from transformers import PretrainedConfig
    from petals.utils.misc import DUMMY

    class MockWordEmbeddings:
        def __init__(self, dtype, device):
            self.weight = unittest.mock.MagicMock()
            self.weight.dtype = dtype
            self.weight.device = device

    config = PretrainedConfig()
    config.tuning_mode = "ptune"
    config.pre_seq_len = 5
    config.hidden_size = 8
    config.num_hidden_layers = 4

    model = get_mock_model(config, MockWordEmbeddings)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts is DUMMY or torch.allclose(intermediate_prompts, DUMMY)

def test_deep_ptune_shapes():
    import torch
    from transformers import PretrainedConfig

    class MockWordEmbeddings:
        def __init__(self, dtype, device):
            self.weight = unittest.mock.MagicMock()
            self.weight.dtype = dtype
            self.weight.device = device

    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5
    config.hidden_size = 8
    config.num_hidden_layers = 4

    model = get_mock_model(config, MockWordEmbeddings)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check regular prompts
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # Check intermediate prompts
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # Check that the first layer's intermediate prompts are all zeros
    assert torch.all(intermediate_prompts[0] == 0)

if __name__ == "__main__":
    pytest.main([__file__])
