import sys
import unittest.mock as mock
import importlib.util
import torch
import torch.nn as nn
from transformers import PretrainedConfig

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

mock_modules = {
    'petals': MockPackage(),
    'petals.utils': MockPackage(),
    'petals.utils.misc': MockPackage(),
    'hivemind': MockPackage(),
}

with mock.patch.dict('sys.modules', mock_modules):
    import petals.utils.misc
    petals.utils.misc.DUMMY = torch.empty(0)

    spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)
    sys.modules["petals.client.ptune"] = ptune
    spec.loader.exec_module(ptune)

class DummyModel(ptune.PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)

def test_deep_ptune():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        num_hidden_layers=3,
        hidden_size=8
    )

    model = DummyModel(config)
    model.init_prompts(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # First layer intermediate prompts should be all zeros
    assert torch.all(intermediate_prompts[0] == 0)

    # Other layers should not be zeros
    assert not torch.all(intermediate_prompts[1:] == 0)

if __name__ == "__main__":
    test_deep_ptune()
