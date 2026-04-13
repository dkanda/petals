import pytest
import torch
import unittest.mock as mock
from transformers import PretrainedConfig

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def setup_module(module):
    # Need to isolate imports so we don't bleed out of this test file
    pass

class DummyModel:
    def __init__(self, config, ptune):
        self.config = config
        self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)

        class MixinWrappedModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)
        self.model = MixinWrappedModel(config)

    def get_prompt(self, batch_size):
        return self.model.get_prompt(batch_size)

def test_deep_ptune():
    with mock.patch.dict('sys.modules', {
        'hivemind': MockPackage(),
        'petals': MockPackage(),
        'petals.utils': mock.MagicMock(),
        'petals.utils.misc': mock.MagicMock(DUMMY=torch.empty(0)),
    }):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["ptune"] = ptune
        spec.loader.exec_module(ptune)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=4
        )

        dummy = DummyModel(config, ptune)
        prompts, intermediate_prompts = dummy.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 16])
        assert intermediate_prompts.shape == torch.Size([4, 2, 5, 16])
        assert torch.all(intermediate_prompts[0] == 0)

def test_ptune():
    with mock.patch.dict('sys.modules', {
        'hivemind': MockPackage(),
        'petals': MockPackage(),
        'petals.utils': mock.MagicMock(),
        'petals.utils.misc': mock.MagicMock(DUMMY=torch.empty(0)),
    }):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["ptune"] = ptune
        spec.loader.exec_module(ptune)

        config = PretrainedConfig(
            tuning_mode="ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=4
        )

        dummy = DummyModel(config, ptune)
        prompts, intermediate_prompts = dummy.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 16])
        # Can check by shape for DUMMY
        assert intermediate_prompts.shape == torch.Size([0])
