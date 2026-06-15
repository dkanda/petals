import sys
import os
import unittest
from unittest import mock
import torch

sys.path.insert(0, os.path.abspath('src'))

class TestPTune(unittest.TestCase):
    def test_deep_ptune_intermediate_prompts_shape(self):
        # We want to load only `src/petals/client/ptune.py` and skip importing `petals` __init__.
        import importlib.util
        spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)

        sys.modules['petals.utils.misc'] = mock.MagicMock()
        sys.modules['petals.utils.misc'].DUMMY = torch.empty(0)
        sys.modules['hivemind'] = mock.MagicMock()

        spec.loader.exec_module(ptune)
        from transformers import PretrainedConfig

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32
                self.init_prompts(config)

        config = PretrainedConfig()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 5
        config.hidden_size = 16
        config.num_hidden_layers = 10

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        self.assertEqual(prompts.shape, torch.Size([2, 5, 16]))
        self.assertEqual(intermediate_prompts.shape, torch.Size([9, 2, 5, 16]))

if __name__ == '__main__':
    unittest.main()
