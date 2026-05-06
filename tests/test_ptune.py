import sys
import unittest.mock
import importlib.util
from unittest import mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig

def test_ptune_shapes():
    class MockPackage:
        pass

    class MockHivemind:
        def get_logger(self, name):
            import logging
            return logging.getLogger(name)

    mock_hivemind = MockHivemind()

    mock_petals = MockPackage()
    mock_petals_utils = MockPackage()
    mock_petals_utils_misc = MockPackage()
    mock_petals_utils_misc.DUMMY = torch.zeros(0)

    spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)

    mock_petals.client = MockPackage()
    mock_petals.client.ptune = ptune

    with mock.patch.dict('sys.modules', {
        'petals': mock_petals,
        'petals.utils': mock_petals_utils,
        'petals.utils.misc': mock_petals_utils_misc,
        'hivemind': mock_hivemind,
        'petals.client.ptune': ptune,
    }):
        spec.loader.exec_module(ptune)

        class DummyConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.pre_seq_len = 10
                self.hidden_size = 64
                self.num_hidden_layers = 12
                self.tuning_mode = "deep_ptune"

        class DummyModel(ptune.PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        with unittest.mock.patch('petals.client.ptune._original_register_parameter', torch.nn.Module.register_parameter):
            config = DummyConfig()
            model = DummyModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)
            assert intermediate_prompts.shape == (config.num_hidden_layers - 1, 2, config.pre_seq_len, config.hidden_size)
