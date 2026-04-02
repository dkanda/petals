import sys
import unittest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

import importlib.util
spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
ptune = importlib.util.module_from_spec(spec)

import unittest.mock
ptune.get_logger = unittest.mock.MagicMock()
ptune.DUMMY = torch.empty(0)

sys.modules["petals.utils.misc"] = unittest.mock.MagicMock()
sys.modules["petals.utils.misc"].DUMMY = ptune.DUMMY
sys.modules["hivemind"] = unittest.mock.MagicMock()

spec.loader.exec_module(ptune)

class DummyModel(nn.Module, ptune.PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

class TestPTune(unittest.TestCase):
    def test_deep_ptune_shapes(self):
        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=4,
        )
        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        self.assertEqual(prompts.shape, torch.Size([2, 5, 16]))
        self.assertEqual(intermediate_prompts.shape, torch.Size([4, 2, 5, 16]))

        # The first layer should be zero-padded
        self.assertTrue(torch.all(intermediate_prompts[0] == 0))

        # Other layers should not be zero-padded natively (unless randomly initialized to zero)
        # Using a small sum to verify they are non-zero mostly
        self.assertFalse(torch.all(intermediate_prompts[1] == 0))

if __name__ == "__main__":
    unittest.main()
