import sys
import unittest
import importlib.util
from unittest.mock import MagicMock, patch

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

mock_modules = {
    'torch': MockPackage(),
    'torch.nn': MockPackage(),
    'hivemind': MockPackage(),
    'transformers': MockPackage(),
}

with patch.dict('sys.modules', mock_modules):
    import torch
    import torch.nn as nn
    torch.float32 = "float32"
    torch.long = "long"

    class MockTensor(MagicMock):
        def __init__(self, shape=None, **kwargs):
            super().__init__(**kwargs)
            self.shape = shape
            self.device = "cpu"
            self.dtype = "float32"

        def unsqueeze(self, *args): return self
        def expand(self, *args): return self
        def to(self, *args): return self
        def view(self, *args):
            return MockTensor(shape=args)
        def permute(self, *args):
            if hasattr(self, 'shape') and self.shape:
                indices = args[0]
                new_shape = tuple(self.shape[i] for i in indices)
                return MockTensor(shape=new_shape)
            return self

    torch.arange = lambda x: MockTensor()

    # We only want to import PTuneMixin, avoiding the rest of petals
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)

    # Mock miscellany imported by ptune
    sys.modules['petals'] = MockPackage()
    sys.modules['petals.utils'] = MockPackage()
    sys.modules['petals.utils.misc'] = MockPackage()

    spec.loader.exec_module(ptune)
    PTuneMixin = ptune.PTuneMixin

class TestConfig:
    tuning_mode = "deep_ptune"
    pre_seq_len = 5
    hidden_size = 16
    num_hidden_layers = 3

class MockModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = MagicMock()
        self.word_embeddings.weight = MockTensor()
        self.init_prompts(config)

class TestPTune(unittest.TestCase):
    def test_deep_ptune_init(self):
        with patch.dict('sys.modules', mock_modules):
            config = TestConfig()
            model = MockModel(config)

            nn.Embedding.assert_any_call(5, 16, dtype="float32")
            nn.Embedding.assert_any_call(5, 2 * 16, dtype="float32") # num_hidden_layers - 1 = 2

    def test_deep_ptune_get_prompt(self):
        with patch.dict('sys.modules', mock_modules):
            config = TestConfig()
            model = MockModel(config)

            model.prompt_embeddings = MagicMock(return_value=MockTensor(shape=(2, 5, 16)))
            model.intermediate_prompt_embeddings = MagicMock(return_value=MockTensor(shape=(2, 5, 2 * 16)))

            with patch('torch.zeros', return_value=MockTensor(shape=(1, 2, 5, 16))) as mock_zeros:
                with patch('torch.cat', return_value=MockTensor(shape=(3, 2, 5, 16))) as mock_cat:
                    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

                    self.assertEqual(prompts.shape, (2, 5, 16))
                    self.assertEqual(intermediate_prompts.shape, (3, 2, 5, 16))

                    mock_zeros.assert_called_once_with(1, 2, 5, 16, device="cpu", dtype="float32")

if __name__ == '__main__':
    unittest.main()
