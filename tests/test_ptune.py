import os
import sys
import importlib.util
from unittest import mock
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('src'))

# Load module directly bypassing __init__ imports
spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
ptune = importlib.util.module_from_spec(spec)

# Mock required imports
hivemind_mock = mock.MagicMock()
transformers_mock = mock.MagicMock()
transformers_mock.PretrainedConfig = type("PretrainedConfig", (), {})

misc_mock = mock.MagicMock()
misc_mock.DUMMY = torch.empty(0)

sys.modules['hivemind'] = hivemind_mock
sys.modules['transformers'] = transformers_mock
sys.modules['petals.utils.misc'] = misc_mock

spec.loader.exec_module(ptune)

def test_ptune_shapes():
    class DummyConfig(transformers_mock.PretrainedConfig):
        def __init__(self):
            self.tuning_mode = "deep_ptune"
            self.pre_seq_len = 5
            self.hidden_size = 8
            self.num_hidden_layers = 4

    class DummyModel(ptune.PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = nn.Embedding(10, config.hidden_size)
            self.init_prompts(config)

    # Fix ptune module's register parameter to standard for tests
    with mock.patch.object(ptune, '_original_register_parameter', nn.Module.register_parameter):
        model = DummyModel(DummyConfig())
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 8])
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 8])
