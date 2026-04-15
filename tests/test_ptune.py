import sys
import unittest.mock as mock
import torch

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

hivemind_mock = MockPackage()
hivemind_mock.get_logger = mock.MagicMock()

petals_mock = MockPackage()
petals_utils_mock = MockPackage()
petals_utils_misc_mock = MockPackage()
petals_utils_misc_mock.DUMMY = torch.empty(0)

# Apply mocks to sys.modules BEFORE importing ptune
with mock.patch.dict('sys.modules', {
    'petals': petals_mock,
    'petals.utils': petals_utils_mock,
    'petals.utils.misc': petals_utils_misc_mock,
    'hivemind': hivemind_mock
}):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ptune)

import torch.nn as nn
from transformers import PretrainedConfig

class MockModel(nn.Module, ptune.PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=5
    )
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 10

    model = MockModel(config)
    batch_size = 4
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # Verify that the first layer's padding is zero
    assert torch.all(intermediate_prompts[0] == 0)

def test_ptune_normal():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=5
    )
    config.tuning_mode = "ptune"
    config.pre_seq_len = 10

    model = MockModel(config)
    batch_size = 4
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (0,)  # DUMMY is empty

if __name__ == "__main__":
    test_deep_ptune()
    test_ptune_normal()
    print("Test passed!")
