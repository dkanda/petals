import sys
import unittest
from unittest import mock
import torch
import torch.nn as nn

sys.path.append('src')

class DummyConfig:
    def __init__(self, tuning_mode=None, pre_seq_len=0, hidden_size=64, num_hidden_layers=3):
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class DummyWordEmbeddings:
    def __init__(self):
        self.weight = mock.MagicMock()
        self.weight.device = torch.device('cpu')
        self.weight.dtype = torch.float32

class TestPTuneMixin(unittest.TestCase):
    def test_init_prompts_deep_ptune(self):
        import types
        mock_hivemind = types.ModuleType("hivemind")
        mock_hivemind.__path__ = []
        mock_hivemind.get_logger = mock.MagicMock()

        sys_modules_patch = {
            'hivemind': mock_hivemind,
            'transformers': mock.MagicMock(),
            'petals.utils.misc': mock.MagicMock(),
            'packaging': mock.MagicMock()
        }

        with mock.patch.dict('sys.modules', sys_modules_patch):
            import importlib.util
            spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
            ptune = importlib.util.module_from_spec(spec)
            ptune.DUMMY = torch.empty(0)

            with mock.patch('builtins.__import__', side_effect=__import__):
                spec.loader.exec_module(ptune)

            class MockMixin(ptune.PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = DummyWordEmbeddings()

            config = DummyConfig(tuning_mode="deep_ptune", pre_seq_len=10, hidden_size=64, num_hidden_layers=3)
            mixin = MockMixin(config)

            mixin.init_prompts(config)

            # The shape should be (pre_seq_len, (num_hidden_layers - 1) * hidden_size)
            self.assertEqual(mixin.intermediate_prompt_embeddings.weight.shape, (10, 2 * 64))

            prompts, intermediate_prompts = mixin.get_prompt(batch_size=2)

            # intermediate_prompts is prepended with prompts, resulting in shape (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
            self.assertEqual(prompts.shape, (2, 10, 64))
            self.assertEqual(intermediate_prompts.shape, (3, 2, 10, 64))

if __name__ == '__main__':
    unittest.main()
