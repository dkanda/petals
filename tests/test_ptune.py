import sys
import importlib.util
from unittest import mock

# Mock deeply to avoid import errors when loading ptune module
class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

sys.modules["hivemind"] = MockPackage()
sys.modules["petals"] = MockPackage()

import torch
from transformers import PretrainedConfig

def test_ptune_shapes():
    # Load module dynamically to inject DUMMY cleanly
    spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
    ptune_module = importlib.util.module_from_spec(spec)
    sys.modules["petals.client.ptune"] = ptune_module

    class DummyMisc:
        DUMMY = torch.empty(0)
    sys.modules["petals.utils.misc"] = DummyMisc()

    spec.loader.exec_module(ptune_module)

    class DummyModel(ptune_module.PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
            self.init_prompts(config)

    config = PretrainedConfig()
    config.tuning_mode = "ptune"
    config.pre_seq_len = 5
    config.hidden_size = 16
    config.num_hidden_layers = 12

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 16)
    assert isinstance(intermediate_prompts, torch.Tensor) and intermediate_prompts.numel() == 0

def test_deep_ptune_shapes():
    # Load module dynamically to inject DUMMY cleanly
    spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
    ptune_module = importlib.util.module_from_spec(spec)
    sys.modules["petals.client.ptune"] = ptune_module

    class DummyMisc:
        DUMMY = torch.empty(0)
    sys.modules["petals.utils.misc"] = DummyMisc()

    spec.loader.exec_module(ptune_module)

    class DummyModel(ptune_module.PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
            self.init_prompts(config)

    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5
    config.hidden_size = 16
    config.num_hidden_layers = 12

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 16)
    # Check that intermediate_prompts has length num_hidden_layers
    assert intermediate_prompts.shape == (12, 2, 5, 16)
    # Check that the first layer is padded with zeros
    assert torch.all(intermediate_prompts[0] == 0)
