import sys
import os

from unittest import mock
import torch

class DummyPetalsMisc:
    DUMMY = torch.empty(0)

class MockHivemind(mock.MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return mock.MagicMock()

mocks = {
    'hivemind': MockHivemind(),
    'petals.utils.misc': DummyPetalsMisc(),
}

with mock.patch.dict('sys.modules', mocks):
    # Load ptune directly from file to avoid __init__ loading everything else
    import importlib.util
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)
    sys.modules["ptune"] = ptune
    spec.loader.exec_module(ptune)

    from transformers import PretrainedConfig

    class DummyModel(ptune.PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = mock.MagicMock()
            self.word_embeddings.weight = torch.zeros(1)
            self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 10
    config.hidden_size = 32
    config.num_hidden_layers = 12

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 10, 32), f"Expected (2, 10, 32), got {prompts.shape}"
    assert intermediate_prompts.shape == (11, 2, 10, 32), f"Expected (11, 2, 10, 32), got {intermediate_prompts.shape}"
