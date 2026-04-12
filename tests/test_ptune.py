import pytest
import sys
import unittest.mock as mock
import importlib.util
import torch
import torch.nn as nn
from transformers import PretrainedConfig

# We use importlib to avoid triggering the entire petals package initialization
# which may require difficult-to-install dependencies in the current env.
# However, we DO NOT mutate sys.modules globally at the top level.

@pytest.fixture
def ptune_module():
    # Provide localized mocks for the test execution context
    with mock.patch.dict('sys.modules', {
        'petals': mock.MagicMock(__path__=[], __spec__=None),
        'petals.utils': mock.MagicMock(),
        'petals.utils.misc': mock.MagicMock(),
        'hivemind': mock.MagicMock(__path__=[], __spec__=None)
    }):
        import petals.utils.misc as misc
        misc.DUMMY = mock.MagicMock()

        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune)
        return ptune

def test_ptune_shapes(ptune_module):
    class MockModel(nn.Module, ptune_module.PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(10, config.hidden_size)
            self.init_prompts(config)

    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=3,
    )
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5

    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 64)
    assert intermediate_prompts.shape == (3, 2, 5, 64)
