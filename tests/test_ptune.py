import torch
import pytest
import sys
import unittest.mock as mock
from transformers import PretrainedConfig

def test_ptune_shapes():
    class MockPackage(mock.MagicMock):
        __path__ = []
        __spec__ = None

    with mock.patch.dict('sys.modules', {
        'petals': MockPackage(),
        'petals.utils': MockPackage(),
        'petals.client': MockPackage(),
        'hivemind': mock.MagicMock(),
    }):
        # We also need to dynamically create a module for petals.utils.misc
        misc_module = type(sys)('petals.utils.misc')
        misc_module.DUMMY = torch.empty(0)

        with mock.patch.dict('sys.modules', {'petals.utils.misc': misc_module}):
            import importlib.util
            spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
            ptune = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ptune)

            PTuneMixin = ptune.PTuneMixin

            class DummyWordEmbeddings:
                def __init__(self, hidden_size):
                    self.weight = torch.empty((100, hidden_size))

            class DummyConfig(PretrainedConfig):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.hidden_size = kwargs.get('hidden_size', 16)
                    self.num_hidden_layers = kwargs.get('num_hidden_layers', 4)
                    self.tuning_mode = kwargs.get('tuning_mode', None)
                    self.pre_seq_len = kwargs.get('pre_seq_len', 0)

            class DummyModel(PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = DummyWordEmbeddings(config.hidden_size)
                    self.init_prompts(config)

            config = DummyConfig(tuning_mode="ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=4)
            model = DummyModel(config)
            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)
            assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
            assert intermediate_prompts is misc_module.DUMMY or intermediate_prompts.numel() == 0

def test_deep_ptune_shapes():
    class MockPackage(mock.MagicMock):
        __path__ = []
        __spec__ = None

    with mock.patch.dict('sys.modules', {
        'petals': MockPackage(),
        'petals.utils': MockPackage(),
        'petals.client': MockPackage(),
        'hivemind': mock.MagicMock(),
    }):
        misc_module = type(sys)('petals.utils.misc')
        misc_module.DUMMY = torch.empty(0)

        with mock.patch.dict('sys.modules', {'petals.utils.misc': misc_module}):
            import importlib.util
            spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
            ptune = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ptune)

            PTuneMixin = ptune.PTuneMixin

            class DummyWordEmbeddings:
                def __init__(self, hidden_size):
                    self.weight = torch.empty((100, hidden_size))

            class DummyConfig(PretrainedConfig):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.hidden_size = kwargs.get('hidden_size', 16)
                    self.num_hidden_layers = kwargs.get('num_hidden_layers', 4)
                    self.tuning_mode = kwargs.get('tuning_mode', None)
                    self.pre_seq_len = kwargs.get('pre_seq_len', 0)

            class DummyModel(PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = DummyWordEmbeddings(config.hidden_size)
                    self.init_prompts(config)

            config = DummyConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=4)
            model = DummyModel(config)
            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)
            assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
            assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)
            assert torch.all(intermediate_prompts[0] == 0)
