import sys
import os
import unittest
from unittest import mock

import torch
from transformers import PretrainedConfig

class TestPTuneMixin(unittest.TestCase):
    def test_ptune_shapes(self):
        # We must isolate the heavy mock additions to sys.modules within this context
        # so they do not pollute the global testing environment.

        hivemind_mock = mock.MagicMock()

        petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
        petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
        petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
        petals_utils_misc_mock = mock.MagicMock(__path__=["src/petals/utils/misc.py"], __spec__=None)
        petals_utils_misc_mock.DUMMY = torch.empty(0)
        petals_utils_mock.misc = petals_utils_misc_mock

        petals_mock.client = petals_client_mock
        petals_mock.utils = petals_utils_mock

        new_modules = {
            'hivemind': hivemind_mock,
            'petals': petals_mock,
            'petals.client': petals_client_mock,
            'petals.utils': petals_utils_mock,
            'petals.utils.misc': petals_utils_misc_mock,
        }

        sys.path.insert(0, os.path.abspath('src'))

        with mock.patch.dict('sys.modules', new_modules):
            import petals.client.ptune as ptune

            class DummyModel(ptune.PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = mock.MagicMock()
                    self.word_embeddings.weight.device = torch.device('cpu')
                    self.word_embeddings.weight.dtype = torch.float32

            config = PretrainedConfig()
            config.tuning_mode = "deep_ptune"
            config.pre_seq_len = 10
            config.hidden_size = 64
            config.num_hidden_layers = 4

            model = DummyModel(config)
            model.init_prompts(config)

            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            self.assertEqual(prompts.shape, torch.Size([2, 10, 64]))
            self.assertEqual(intermediate_prompts.shape, torch.Size([3, 2, 10, 64]))

            if hasattr(model, 'intermediate_prompt_embeddings'):
                self.assertEqual(model.intermediate_prompt_embeddings.weight.shape, torch.Size([10, 192]))

        sys.path.pop(0)

if __name__ == "__main__":
    unittest.main()
