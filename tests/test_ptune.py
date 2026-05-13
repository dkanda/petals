import sys
import unittest
from unittest import mock

class DummyConfig:
    def __init__(self):
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 10
        self.hidden_size = 32
        self.num_hidden_layers = 4

class DummyWordEmbeddings:
    def __init__(self):
        self.weight = mock.MagicMock()
        self.weight.device = "cpu"
        self.weight.dtype = mock.MagicMock()

class TestPTuneMixin(unittest.TestCase):
    def test_init_prompts_and_get_prompt_deep_ptune(self):
        import types
        mocks = {
            'packaging': types.ModuleType('packaging'),
            'packaging.version': types.ModuleType('packaging.version'),
        }

        def get_mock(name):
            if name not in mocks:
                m = mock.MagicMock()
                m.__path__ = []
                m.__spec__ = None
                mocks[name] = m
            return mocks[name]

        with mock.patch.dict('sys.modules', {
            'torch': get_mock('torch'),
            'torch.nn': get_mock('torch.nn'),
            'hivemind': get_mock('hivemind'),
            'transformers': get_mock('transformers'),
            'packaging': mocks['packaging'],
            'packaging.version': mocks['packaging.version'],
            'petals': get_mock('petals'),
            'petals.utils.misc': get_mock('petals.utils.misc'),
        }):
            import importlib.util
            spec = importlib.util.spec_from_file_location("petals_ptune", "src/petals/client/ptune.py")
            ptune = importlib.util.module_from_spec(spec)
            sys.modules["petals_ptune"] = ptune
            spec.loader.exec_module(ptune)
            PTuneMixin = ptune.PTuneMixin

            import torch
            import torch.nn as nn

            with mock.patch('petals_ptune.force_non_empty_weights'):
                mixin = PTuneMixin()
                config = DummyConfig()
                mixin.word_embeddings = DummyWordEmbeddings()
                mixin.config = config

                mixin.init_prompts(config)

                # Check if intermediate_prompt_embeddings was initialized with (num_hidden_layers - 1)
                expected_size = (config.num_hidden_layers - 1) * config.hidden_size
                nn.Embedding.assert_any_call(config.pre_seq_len, expected_size, dtype=torch.float32)

                # Test get_prompt
                mixin.get_prompt(batch_size=2)

                # Check if torch.cat was called to prepend prompts
                self.assertTrue(torch.cat.called)

if __name__ == '__main__':
    unittest.main()
