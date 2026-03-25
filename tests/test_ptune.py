import sys
import unittest
from unittest.mock import MagicMock, patch

# Create a mock for hivemind and its subpackages
class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

mock_modules = {
    'hivemind': MockPackage(),
    'hivemind.utils': MockPackage(),
    'hivemind.moe': MockPackage(),
    'hivemind.p2p': MockPackage(),
    'hivemind.compression': MockPackage(),
    'tensor_parallel': MockPackage(),
    'tensor_parallel.tensor_parallel': MockPackage(),
    'tensor_parallel.slicing_configs': MockPackage(),
    'petals.client.inference_session': MockPackage(),
    'petals.client.remote_generation': MockPackage(),
    'petals.client.remote_forward_backward': MockPackage(),
    'petals.client.routing': MockPackage(),
    'petals.models': MockPackage(),
    'petals.models.bloom': MockPackage(),
    'petals.models.llama': MockPackage(),
    'petals.models.mixtral': MockPackage(),
    'petals.models.falcon': MockPackage(),
    'petals.models.deepseek': MockPackage(),
    'petals.client.from_pretrained': MockPackage(),
    'petals.server.memory_cache': MockPackage(),
    'petals.server.handler': MockPackage(),
}

patcher = patch.dict('sys.modules', mock_modules)
patcher.start()

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin, PTuneConfig
from petals.utils.misc import DUMMY

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

class TestPTuneMixin(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        patcher.stop()
    def test_ptune_init(self):
        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=4,
            pre_seq_len=8,
            tuning_mode="ptune"
        )
        model = DummyModel(config)

        self.assertTrue(hasattr(model, 'prompt_embeddings'))
        self.assertFalse(hasattr(model, 'intermediate_prompt_embeddings'))
        self.assertEqual(model.prompt_embeddings.weight.shape, (8, 64))

    def test_deep_ptune_init(self):
        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=4,
            pre_seq_len=8,
            tuning_mode="deep_ptune"
        )
        model = DummyModel(config)

        self.assertTrue(hasattr(model, 'prompt_embeddings'))
        self.assertTrue(hasattr(model, 'intermediate_prompt_embeddings'))
        self.assertEqual(model.prompt_embeddings.weight.shape, (8, 64))
        # num_hidden_layers - 1
        self.assertEqual(model.intermediate_prompt_embeddings.weight.shape, (8, 3 * 64))

    def test_get_prompt_ptune(self):
        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=4,
            pre_seq_len=8,
            tuning_mode="ptune"
        )
        model = DummyModel(config)
        batch_size = 2

        prompts, intermediate_prompts = model.get_prompt(batch_size)

        self.assertEqual(prompts.shape, (batch_size, 8, 64))
        self.assertIs(intermediate_prompts, DUMMY)

    def test_get_prompt_deep_ptune(self):
        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=4,
            pre_seq_len=8,
            tuning_mode="deep_ptune"
        )
        model = DummyModel(config)
        batch_size = 2

        prompts, intermediate_prompts = model.get_prompt(batch_size)

        self.assertEqual(prompts.shape, (batch_size, 8, 64))
        # Should have shape (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
        self.assertEqual(intermediate_prompts.shape, (4, batch_size, 8, 64))

        # Verify first layer is padded with zeros
        first_layer = intermediate_prompts[0]
        self.assertTrue(torch.all(first_layer == 0))

        # Verify other layers are not completely zeros (they come from embeddings)
        # Note: embeddings are initialized randomly, very small chance of being all zeros
        # but just in case, we check it's not strictly equal to the first layer padding
        for i in range(1, 4):
            self.assertFalse(torch.all(intermediate_prompts[i] == 0))

if __name__ == '__main__':
    unittest.main()
