import sys
import types
import unittest.mock
import logging
import importlib.util
import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

def create_mock_package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module

def test_ptune_init_and_get_prompt():
    mock_hivemind = create_mock_package('hivemind')
    mock_hivemind.get_logger = logging.getLogger

    # Mock petals to prevent deep imports
    mock_petals = create_mock_package('petals')
    mock_petals_utils = create_mock_package('petals.utils')
    mock_petals_utils_misc = create_mock_package('petals.utils.misc')
    mock_petals_utils_misc.DUMMY = None

    modules_to_patch = {
        'hivemind': mock_hivemind,
        'petals': mock_petals,
        'petals.utils': mock_petals_utils,
        'petals.utils.misc': mock_petals_utils_misc
    }

    with unittest.mock.patch.dict('sys.modules', modules_to_patch):
        spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune)

    class DummyConfig(PretrainedConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.pre_seq_len = 10
            self.hidden_size = 16
            self.num_hidden_layers = 4
            self.tuning_mode = "deep_ptune"

    class DummyModel(nn.Module, ptune.PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

    config = DummyConfig()

    with unittest.mock.patch.object(ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
        model = DummyModel(config)

    assert model.pre_seq_len == 10

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, 10, 16)
    assert intermediate_prompts.shape == (4, batch_size, 10, 16)
