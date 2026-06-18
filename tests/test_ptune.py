import sys
import os
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

import torch
import torch.nn as nn
from transformers import PretrainedConfig

# Mock hivemind to avoid ModuleNotFoundError
with mock.patch.dict('sys.modules', {
    'petals.utils.misc': mock.MagicMock(),
    'hivemind': mock.MagicMock()
}):
    # Import just the ptune module, not the full petals package to bypass unresolvable init dependencies locally
    import importlib.util
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)
    sys.modules["ptune"] = ptune
    spec.loader.exec_module(ptune)

class TestPTuneMixin(unittest.TestCase):
    def test_deep_ptune_intermediate_shape(self):
        config = PretrainedConfig()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 5
        config.num_hidden_layers = 12
        config.hidden_size = 64

        class DummyModel(nn.Module, ptune.PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        model = DummyModel(config)

        # Test shape of intermediate prompt embeddings
        # Expected: pre_seq_len x ((num_hidden_layers - 1) * hidden_size)
        expected_size = (config.num_hidden_layers - 1) * config.hidden_size
        self.assertEqual(model.intermediate_prompt_embeddings.weight.shape, (config.pre_seq_len, expected_size))

        # Test get_prompt output
        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        # prompts shape: batch_size x pre_seq_len x hidden_size
        self.assertEqual(prompts.shape, (batch_size, config.pre_seq_len, config.hidden_size))

        # intermediate_prompts shape: (num_hidden_layers - 1) x batch_size x pre_seq_len x hidden_size
        self.assertEqual(
            intermediate_prompts.shape,
            (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
        )

if __name__ == '__main__':
    unittest.main()
