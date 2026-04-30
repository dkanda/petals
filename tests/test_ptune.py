import sys
import types
from unittest import mock
import pytest
import importlib.util
import torch
import torch.nn as nn

def test_ptune_intermediate_embeddings():
    # Setup global mocks for required dependencies to run in isolated test env
    mock_modules = {}

    mock_hivemind = types.ModuleType("hivemind")
    mock_hivemind.__path__ = []
    mock_hivemind.get_logger = mock.MagicMock()
    mock_modules["hivemind"] = mock_hivemind

    mock_transformers = types.ModuleType("transformers")
    mock_transformers.__path__ = []
    class PretrainedConfig:
        pass
    mock_transformers.PretrainedConfig = PretrainedConfig
    mock_modules["transformers"] = mock_transformers

    mock_petals = types.ModuleType("petals")
    mock_petals.__path__ = []
    mock_modules["petals"] = mock_petals

    mock_petals_utils = types.ModuleType("petals.utils")
    mock_petals_utils.__path__ = []
    mock_modules["petals.utils"] = mock_petals_utils

    mock_petals_utils_misc = types.ModuleType("petals.utils.misc")
    DUMMY = torch.zeros(1)
    mock_petals_utils_misc.DUMMY = DUMMY
    mock_modules["petals.utils.misc"] = mock_petals_utils_misc

    with mock.patch.dict(sys.modules, mock_modules):
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

    # Recreate the context and model structure
    class Config(ptune.PretrainedConfig):
        def __init__(self):
            self.tuning_mode = "deep_ptune"
            self.pre_seq_len = 5
            self.hidden_size = 16
            self.num_hidden_layers = 4

    class Model(nn.Module, ptune.PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(10, config.hidden_size)
            self.init_prompts(config)

    config = Config()
    model = Model(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, 5, 16)
    assert intermediate_prompts.shape == (4, batch_size, 5, 16)
    assert torch.all(intermediate_prompts[0] == 0)

def test_ptune_standard():
    mock_modules = {}

    mock_hivemind = types.ModuleType("hivemind")
    mock_hivemind.__path__ = []
    mock_hivemind.get_logger = mock.MagicMock()
    mock_modules["hivemind"] = mock_hivemind

    mock_transformers = types.ModuleType("transformers")
    mock_transformers.__path__ = []
    class PretrainedConfig:
        pass
    mock_transformers.PretrainedConfig = PretrainedConfig
    mock_modules["transformers"] = mock_transformers

    mock_petals = types.ModuleType("petals")
    mock_petals.__path__ = []
    mock_modules["petals"] = mock_petals

    mock_petals_utils = types.ModuleType("petals.utils")
    mock_petals_utils.__path__ = []
    mock_modules["petals.utils"] = mock_petals_utils

    mock_petals_utils_misc = types.ModuleType("petals.utils.misc")
    DUMMY = torch.zeros(1)
    mock_petals_utils_misc.DUMMY = DUMMY
    mock_modules["petals.utils.misc"] = mock_petals_utils_misc

    with mock.patch.dict(sys.modules, mock_modules):
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

    class Config(ptune.PretrainedConfig):
        def __init__(self):
            self.tuning_mode = "ptune"
            self.pre_seq_len = 5
            self.hidden_size = 16
            self.num_hidden_layers = 4

    class Model(nn.Module, ptune.PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(10, config.hidden_size)
            self.init_prompts(config)

    config = Config()
    model = Model(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, 5, 16)
    assert intermediate_prompts is ptune.DUMMY
