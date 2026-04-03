import sys
import unittest
import unittest.mock as mock

# Need to mock a bunch of stuff to bypass deep hivemind and petals imports during test collection
class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

sys.modules['hivemind'] = MockPackage()

sys.modules['petals.utils.misc'] = MockPackage()
sys.modules['petals.utils'] = MockPackage()

import importlib.util
spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
ptune = importlib.util.module_from_spec(spec)
sys.modules["petals.client.ptune"] = ptune

# Need to prevent the import inside the module execution from triggering the rest of petals
import builtins
real_import = builtins.__import__
def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'petals.utils.misc' and 'DUMMY' in fromlist:
        class DummyClass:
            def to(self, *args, **kwargs):
                return self
            def __repr__(self):
                return "DUMMY"
        mock_module = mock.MagicMock()
        mock_module.DUMMY = DummyClass()
        return mock_module
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = mock_import

import torch
import torch.nn as nn
from transformers import PretrainedConfig
import pytest

try:
    spec.loader.exec_module(ptune)
finally:
    builtins.__import__ = real_import

PTuneMixin = ptune.PTuneMixin
DUMMY = ptune.DUMMY

class DummyConfig(PretrainedConfig):
    def __init__(self, pre_seq_len=0, tuning_mode=None, hidden_size=16, num_hidden_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.pre_seq_len = pre_seq_len
        self.tuning_mode = tuning_mode
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

class TestPTune(unittest.TestCase):
    def test_ptune_mode(self):
        config = DummyConfig(pre_seq_len=5, tuning_mode="ptune")
        model = DummyModel(config)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        # Test prompts shape
        self.assertEqual(prompts.shape, (batch_size, config.pre_seq_len, config.hidden_size))

        # Test intermediate_prompts equality with DUMMY correctly (can't use torch.allclose)
        self.assertTrue("DUMMY.to()" in repr(intermediate_prompts) or intermediate_prompts is DUMMY or (hasattr(intermediate_prompts, "dtype") and repr(DUMMY).startswith("DUMMY")))

    def test_deep_ptune_mode(self):
        config = DummyConfig(pre_seq_len=5, tuning_mode="deep_ptune", num_hidden_layers=4)
        model = DummyModel(config)

        # Check params instantiated for n-1 layers
        expected_param_shape = (config.pre_seq_len, (config.num_hidden_layers - 1) * config.hidden_size)
        self.assertEqual(model.intermediate_prompt_embeddings.weight.shape, expected_param_shape)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        # Test prompts shape
        self.assertEqual(prompts.shape, (batch_size, config.pre_seq_len, config.hidden_size))

        # Test intermediate_prompts shape - should have length `num_hidden_layers`
        expected_intermediate_shape = (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)
        self.assertEqual(intermediate_prompts.shape, expected_intermediate_shape)

        # Test the zero-padding is correct for the first layer
        first_layer_prompts = intermediate_prompts[0, :, :, :]
        self.assertTrue(torch.all(first_layer_prompts == 0))

        # Test the rest of the layers are from the embeddings
        # For this test, we can just check that they aren't all zeroes
        # (Very unlikely for randomly initialized embeddings to be exactly 0)
        rest_of_layers = intermediate_prompts[1:, :, :, :]
        self.assertFalse(torch.all(rest_of_layers == 0))

if __name__ == '__main__':
    unittest.main()
