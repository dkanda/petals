import sys
import os
import unittest
from unittest import mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig

class TestPTune(unittest.TestCase):
    def test_ptune_intermediate_prompts_shape(self):
        sys.path.insert(0, os.path.abspath('src'))
        import importlib.util

        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune_module = importlib.util.module_from_spec(spec)

        class DummyMod:
            DUMMY = torch.empty(0)

        with mock.patch.dict('sys.modules', {'petals.utils.misc': DummyMod(), 'hivemind': mock.MagicMock()}):
            spec.loader.exec_module(ptune_module)

            class TestModel(nn.Module, ptune_module.PTuneMixin):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = nn.Embedding(100, config.hidden_size)
                    self.init_prompts(config)

            config = PretrainedConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=3)
            model = TestModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            self.assertEqual(prompts.shape, torch.Size([2, 5, 16]))
            self.assertEqual(intermediate_prompts.shape, torch.Size([2, 2, 5, 16]))
