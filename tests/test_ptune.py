import unittest
from unittest.mock import patch
import sys
import importlib.util
import types

class MockPackage(types.ModuleType):
    __path__ = []

class TestPTune(unittest.TestCase):
    def test_deep_ptune_shapes(self):
        import torch
        import torch.nn as nn
        from transformers import PretrainedConfig

        # Create mocks
        petals_mock = MockPackage('petals')
        petals_utils_mock = MockPackage('petals.utils')
        petals_utils_misc_mock = MockPackage('petals.utils.misc')
        petals_utils_misc_mock.DUMMY = None
        hivemind_mock = MockPackage('hivemind')
        hivemind_mock.get_logger = lambda name: None
        petals_client_mock = MockPackage('petals.client')

        mocked_modules = {
            'petals': petals_mock,
            'petals.utils': petals_utils_mock,
            'petals.utils.misc': petals_utils_misc_mock,
            'hivemind': hivemind_mock,
            'petals.client': petals_client_mock,
        }

        with patch.dict('sys.modules', mocked_modules):
            spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
            ptune_module = importlib.util.module_from_spec(spec)
            sys.modules["petals.client.ptune"] = ptune_module
            spec.loader.exec_module(ptune_module)

            PTuneMixin = ptune_module.PTuneMixin

            class DummyWordEmbeddings:
                def __init__(self, hidden_size):
                    self.weight = torch.zeros(1, hidden_size)

            class DummyModel(PTuneMixin, nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = DummyWordEmbeddings(config.hidden_size)
                    self.init_prompts(config)

            with patch.object(ptune_module.nn.Module, 'register_parameter', ptune_module._original_register_parameter):
                config = PretrainedConfig()
                config.tuning_mode = "deep_ptune"
                config.pre_seq_len = 5
                config.hidden_size = 16
                config.num_hidden_layers = 4

                model = DummyModel(config)
                batch_size = 2

                prompts, intermediate_prompts = model.get_prompt(batch_size)

                self.assertEqual(prompts.shape, (batch_size, config.pre_seq_len, config.hidden_size))
                self.assertEqual(intermediate_prompts.shape, (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size))

if __name__ == '__main__':
    unittest.main()
