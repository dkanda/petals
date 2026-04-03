import sys
from unittest import mock
import importlib.util

# Setup mocks to avoid deeply loading hivemind during testing
class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

sys.modules['petals'] = MockPackage() # Prevent petals.__init__.py from loading everything

import torch
import torch.nn as nn
from transformers import PretrainedConfig

sys.path.insert(0, "src")

# Directly load the module we want to test to bypass petals.__init__.py
spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
ptune = importlib.util.module_from_spec(spec)
# mock petals.utils.misc.DUMMY inside ptune
class DummyMisc:
    # Use MagicMock so we can call .to(dtype) on it without errors
    DUMMY = mock.MagicMock()
sys.modules['petals.utils.misc'] = DummyMisc
sys.modules['petals.utils'] = MockPackage()

sys.modules['hivemind'] = MockPackage()
sys.modules['hivemind.utils'] = MockPackage()
sys.modules['hivemind.utils.logging'] = MockPackage()

spec.loader.exec_module(ptune)
PTuneMixin = ptune.PTuneMixin

class MockModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    config = PretrainedConfig(hidden_size=64, num_hidden_layers=4, pre_seq_len=8)
    config.tuning_mode="ptune"
    model = MockModel(config)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 8, 64)

def test_deep_ptune_shapes():
    config = PretrainedConfig(hidden_size=64, num_hidden_layers=4, pre_seq_len=8)
    config.tuning_mode="deep_ptune"
    model = MockModel(config)

    # intermediate_prompt_embeddings should have (num_hidden_layers - 1) * hidden_size as embedding dim
    assert model.intermediate_prompt_embeddings.weight.shape[1] == (4 - 1) * 64

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 8, 64)

    # intermediate_prompts should be padded and have length num_hidden_layers
    assert intermediate_prompts.shape == (4, 2, 8, 64)

    # Check that first layer padding is all zeros
    assert torch.all(intermediate_prompts[0] == 0)

    # Check that remaining layers are not all zeros (as they are randomly initialized embeddings)
    assert not torch.all(intermediate_prompts[1:] == 0)
