import unittest
from unittest import mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import sys

# To run this module completely independently of the rest of the project and avoid all deep circular imports and dependency failures,
# we mock the top-level 'petals' package before importing from our source file.
class MockPackage:
    pass
sys.modules['petals'] = MockPackage()

import importlib.util

def load_ptune():
    spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)
    sys.modules["petals.client.ptune"] = ptune

    # We must patch sys.modules to mock hivemind within the execution context of the module to bypass its imports
    with mock.patch.dict('sys.modules', {'hivemind': mock.MagicMock(), 'petals.utils.misc': mock.MagicMock()}):
        # We need a dummy DUMMY
        sys.modules['petals.utils.misc'].DUMMY = torch.tensor(0)
        spec.loader.exec_module(ptune)
    return ptune

class TestPTune(unittest.TestCase):
    def test_ptune_deep(self):
        # apply mock for _original_register_parameter
        with mock.patch('torch.nn.Module.register_parameter', torch.nn.Module.register_parameter):
            ptune = load_ptune()

            class DummyConfig(PretrainedConfig):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.tuning_mode = "deep_ptune"
                    self.pre_seq_len = 5
                    self.hidden_size = 16
                    self.num_hidden_layers = 4

            class DummyModel(ptune.PTuneMixin, nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            config = DummyConfig()
            model = DummyModel(config)

            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            # Since num_hidden_layers is 4, intermediate_prompts should have shape (3, 2, 5, 16)
            self.assertEqual(intermediate_prompts.shape, (3, 2, 5, 16))

if __name__ == '__main__':
    unittest.main()
