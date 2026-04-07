import sys
import unittest.mock as mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import pytest

# We use patch.dict to mock sys.modules dynamically within tests to avoid poisoning pytest cache
@pytest.fixture
def mocked_petals():
    mock_dict = {}
    class MockPackage(mock.MagicMock):
        __path__ = []
        __spec__ = None

    mock_dict['hivemind'] = MockPackage()
    mock_dict['hivemind.p2p'] = MockPackage()
    mock_dict['hivemind.p2p.p2p_daemon_bindings'] = MockPackage()
    mock_dict['hivemind.p2p.p2p_daemon_bindings.datastructures'] = MockPackage()
    mock_dict['hivemind.moe'] = MockPackage()
    mock_dict['hivemind.moe.client'] = MockPackage()
    mock_dict['hivemind.moe.client.remote_expert_worker'] = MockPackage()
    mock_dict['hivemind.moe.server'] = MockPackage()
    mock_dict['hivemind.moe.server.module_backend'] = MockPackage()
    mock_dict['hivemind.moe.expert_uid'] = MockPackage()
    mock_dict['hivemind.utils'] = MockPackage()
    mock_dict['hivemind.utils.nested'] = MockPackage()
    mock_dict['hivemind.utils.asyncio'] = MockPackage()
    mock_dict['hivemind.utils.mpfuture'] = MockPackage()
    mock_dict['hivemind.utils.tensor_descr'] = MockPackage()
    mock_dict['hivemind.dht'] = MockPackage()
    mock_dict['hivemind.dht.routing'] = MockPackage()
    mock_dict['hivemind.dht.node'] = MockPackage()
    mock_dict['hivemind.p2p.p2p_daemon'] = MockPackage()
    mock_dict['hivemind.proto'] = MockPackage()
    mock_dict['hivemind.utils.streaming'] = MockPackage()
    mock_dict['tensor_parallel'] = MockPackage()
    mock_dict['tensor_parallel.slicing_configs'] = MockPackage()
    mock_dict['petals.constants'] = MockPackage()
    mock_dict['petals.data_structures'] = MockPackage()

    def get_logger(name):
        return mock.MagicMock()
    mock_dict['hivemind'].get_logger = get_logger

    class MiscMock(mock.MagicMock):
        DUMMY = torch.empty(0)
    mock_dict['petals.utils'] = MockPackage()
    mock_dict['petals.utils.misc'] = MiscMock()

    with mock.patch.dict('sys.modules', mock_dict):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune)
        yield ptune

def test_ptune_initialization(mocked_petals):
    ptune = mocked_petals
    class DummyConfig(PretrainedConfig, ptune.PTuneConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = 16
            self.num_hidden_layers = 4

    class DummyModel(nn.Module, ptune.PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

    config = DummyConfig(pre_seq_len=5, tuning_mode="ptune")
    model = DummyModel(config)
    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")
    assert model.prompt_embeddings.weight.shape == (5, 16)

def test_deep_ptune_initialization(mocked_petals):
    ptune = mocked_petals
    class DummyConfig(PretrainedConfig, ptune.PTuneConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = 16
            self.num_hidden_layers = 4

    class DummyModel(nn.Module, ptune.PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

    config = DummyConfig(pre_seq_len=5, tuning_mode="deep_ptune")
    model = DummyModel(config)
    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")
    assert model.prompt_embeddings.weight.shape == (5, 16)
    assert model.intermediate_prompt_embeddings.weight.shape == (5, (4 - 1) * 16)

def test_ptune_get_prompt(mocked_petals):
    ptune = mocked_petals
    class DummyConfig(PretrainedConfig, ptune.PTuneConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = 16
            self.num_hidden_layers = 4

    class DummyModel(nn.Module, ptune.PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

    config = DummyConfig(pre_seq_len=5, tuning_mode="ptune")
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    # Check DUMMY handling
    assert "DUMMY" in str(intermediate_prompts) or (isinstance(intermediate_prompts, torch.Tensor) and intermediate_prompts.numel() == 0) or intermediate_prompts is ptune.DUMMY

def test_deep_ptune_get_prompt(mocked_petals):
    ptune = mocked_petals
    class DummyConfig(PretrainedConfig, ptune.PTuneConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = 16
            self.num_hidden_layers = 4

    class DummyModel(nn.Module, ptune.PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

    config = DummyConfig(pre_seq_len=5, tuning_mode="deep_ptune")
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (4, 2, 5, 16)

    # Check that the first layer's intermediate prompt is zero
    assert torch.all(intermediate_prompts[0] == 0)

    # Check that subsequent layers are not necessarily zero
    # (since they are randomly initialized embeddings)
    # We just ensure they are part of the computation graph properly.
    assert intermediate_prompts.device == model.word_embeddings.weight.device
    assert intermediate_prompts.dtype == model.word_embeddings.weight.dtype
