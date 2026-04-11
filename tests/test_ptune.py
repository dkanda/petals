import unittest
from unittest import mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

sys_modules_patch = {
    'petals': MockPackage(),
    'petals.utils': MockPackage(),
    'petals.utils.misc': MockPackage(),
    'hivemind': MockPackage(),
    'hivemind.get_logger': mock.MagicMock()
}

class TestPTuneMixin(unittest.TestCase):
    def test_deep_ptune_get_prompt(self):
        with mock.patch.dict('sys.modules', sys_modules_patch):
            import sys
            import importlib.util

            # Use importlib to bypass top-level __init__ loading
            spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules["petals.client.ptune"] = module
            spec.loader.exec_module(module)

            PTuneMixin = module.PTuneMixin

            # Setup mock model
            class DummyModel(PTuneMixin, nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    # To satisfy dtype and device access
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            config = PretrainedConfig(
                hidden_size=16,
                num_hidden_layers=5,
                tuning_mode="deep_ptune",
                pre_seq_len=8
            )

            model = DummyModel(config)

            # The expected intermediate prompt shape is (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            self.assertEqual(prompts.shape, (batch_size, config.pre_seq_len, config.hidden_size))
            self.assertEqual(
                intermediate_prompts.shape,
                (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)
            )

            # First element of intermediate_prompts should be all zeros
            self.assertTrue(torch.all(intermediate_prompts[0] == 0))

    def test_ptune_get_prompt(self):
        with mock.patch.dict('sys.modules', sys_modules_patch):
            import sys
            import importlib.util

            # Use importlib to bypass top-level __init__ loading
            spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules["petals.client.ptune"] = module
            spec.loader.exec_module(module)

            PTuneMixin = module.PTuneMixin
            from petals.utils.misc import DUMMY

            # Setup mock model
            class DummyModel(PTuneMixin, nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    # To satisfy dtype and device access
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            config = PretrainedConfig(
                hidden_size=16,
                num_hidden_layers=5,
                tuning_mode="ptune",
                pre_seq_len=8
            )

            model = DummyModel(config)

            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            self.assertEqual(prompts.shape, (batch_size, config.pre_seq_len, config.hidden_size))
            self.assertTrue("DUMMY.to()" in repr(intermediate_prompts) or intermediate_prompts is DUMMY or (isinstance(intermediate_prompts, torch.Tensor) and intermediate_prompts.numel() == 0))

if __name__ == '__main__':
    unittest.main()
