import sys
import importlib.util
from unittest import mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def get_isolated_ptune_module():
    mock_modules = {
        'hivemind': MockPackage(),
        'tensor_parallel': MockPackage(),
        'petals': MockPackage(),
        'petals.utils': MockPackage(),
    }

    # We need a real DUMMY for tests
    misc_module = type(sys)("petals.utils.misc")
    misc_module.DUMMY = torch.empty(0)
    mock_modules["petals.utils.misc"] = misc_module

    with mock.patch.dict('sys.modules', mock_modules):
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune_module = importlib.util.module_from_spec(spec)
        # Add to sys.modules specifically for inner imports within the file
        with mock.patch.dict('sys.modules', {"petals.client.ptune": ptune_module}):
            spec.loader.exec_module(ptune_module)
            return ptune_module, misc_module.DUMMY

def test_ptune_shapes():
    ptune_module, real_dummy = get_isolated_ptune_module()

    class MockModel(ptune_module.PTuneMixin, nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )
    model = MockModel(config)
    batch_size = 3
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert "DUMMY.to()" in repr(intermediate_prompts) or intermediate_prompts is real_dummy

def test_deep_ptune_shapes():
    ptune_module, _ = get_isolated_ptune_module()

    class MockModel(ptune_module.PTuneMixin, nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )
    model = MockModel(config)
    batch_size = 3
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # Check that the first layer's padding is zero
    assert torch.all(intermediate_prompts[0] == 0)

if __name__ == "__main__":
    test_ptune_shapes()
    test_deep_ptune_shapes()
    print("All tests passed.")
