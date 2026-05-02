import sys
import types
from unittest import mock

# Mock hivemind
hivemind_mock = types.ModuleType('hivemind')
hivemind_mock.get_logger = mock.MagicMock()
sys.modules['hivemind'] = hivemind_mock

import torch
import torch.nn as nn
from transformers import PretrainedConfig

DUMMY = torch.empty(0)

import importlib.util

def create_module_with_content(filepath, name):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

petals_mock = types.ModuleType('petals')
petals_utils_mock = types.ModuleType('petals.utils')
petals_utils_misc_mock = types.ModuleType('petals.utils.misc')
petals_utils_misc_mock.DUMMY = DUMMY

sys.modules['petals'] = petals_mock
sys.modules['petals.utils'] = petals_utils_mock
sys.modules['petals.utils.misc'] = petals_utils_misc_mock

ptune_module = create_module_with_content("src/petals/client/ptune.py", "petals.client.ptune")

class DummyModel(ptune_module.PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = mock.MagicMock()
        self.word_embeddings.weight.device = torch.device('cpu')
        self.word_embeddings.weight.dtype = torch.float32
        self.init_prompts(config)

def test_deep_ptune():
    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5
    config.hidden_size = 16
    config.num_hidden_layers = 4

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(2)

    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (3, 2, 5, 16)

def test_ptune():
    config = PretrainedConfig()
    config.tuning_mode = "ptune"
    config.pre_seq_len = 5
    config.hidden_size = 16
    config.num_hidden_layers = 4

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(2)

    assert prompts.shape == (2, 5, 16)
    assert "DUMMY.to()" in repr(intermediate_prompts) or intermediate_prompts is DUMMY or (isinstance(intermediate_prompts, torch.Tensor) and intermediate_prompts.numel() == 0)

if __name__ == "__main__":
    test_deep_ptune()
    test_ptune()
