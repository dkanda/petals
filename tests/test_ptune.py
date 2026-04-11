import sys
import unittest.mock as mock
import importlib.util
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import pytest

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def setup_module():
    global ptune

    # Store original modules to restore later
    global original_hivemind, original_petals_misc
    original_hivemind = sys.modules.get('hivemind')
    original_petals_misc = sys.modules.get('petals.utils.misc')

    # Mock necessary dependencies
    sys.modules['hivemind'] = MockPackage()
    sys.modules['petals.utils.misc'] = MockPackage()
    sys.modules['hivemind'].get_logger = mock.MagicMock()
    sys.modules['petals.utils.misc'].DUMMY = torch.zeros(1)

    # Load ptune directly to avoid cascaded imports
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)
    sys.modules["ptune"] = ptune
    spec.loader.exec_module(ptune)

def teardown_module():
    # Restore original modules
    if original_hivemind is not None:
        sys.modules['hivemind'] = original_hivemind
    else:
        del sys.modules['hivemind']

    if original_petals_misc is not None:
        sys.modules['petals.utils.misc'] = original_petals_misc
    else:
        del sys.modules['petals.utils.misc']

    if "ptune" in sys.modules:
        del sys.modules["ptune"]


def test_deep_ptune_shapes():
    class DummyModel(ptune.PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = nn.Embedding(10, config.hidden_size)
            self.init_prompts(config)

    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        num_hidden_layers=3,
        hidden_size=4
    )

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    # Check shapes
    assert prompts.shape == (2, 5, 4), f"Wrong prompts shape: {prompts.shape}"
    assert intermediate_prompts.shape == (3, 2, 5, 4), f"Wrong intermediate prompts shape: {intermediate_prompts.shape}"

    # Check padding is zero
    assert torch.all(intermediate_prompts[0] == 0), "Padding layer should be zero"

    # Check intermediate prompts are not zero (randomly initialized embeddings)
    assert not torch.all(intermediate_prompts[1:] == 0), "Intermediate prompts should not be all zero"
