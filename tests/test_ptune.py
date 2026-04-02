import sys

# Create our own local mock instances only when needed during test runtime
class DummyModelBase:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = type('obj', (object,), {'weight': type('obj', (object,), {'device': 'cpu', 'dtype': torch.float32})})()

class MockPtuneMixin:
    pass

import torch
import torch.nn as nn
from transformers import PretrainedConfig

# Rather than importing from petals, which evaluates everything inside `src/petals` and breaks due to missing local deps,
# we will just import the file explicitly without importing the rest of the package.
import importlib.util
spec = importlib.util.spec_from_file_location("ptune_module", "src/petals/client/ptune.py")
ptune_module = importlib.util.module_from_spec(spec)

class MockLogger:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass

def _get_logger(*args, **kwargs):
    return MockLogger()

import unittest.mock
def setup_module():
    global patcher
    mock_modules = {
        'petals': type('obj', (object,), {'__path__': []})(),
        'petals.utils': type('obj', (object,), {'__path__': []})(),
        'petals.utils.misc': type('obj', (object,), {'DUMMY': torch.empty(0)})(),
        'hivemind': type('obj', (object,), {'get_logger': _get_logger})()
    }
    patcher = unittest.mock.patch.dict(sys.modules, mock_modules)
    patcher.start()

    spec.loader.exec_module(ptune_module)
    global PTuneMixin
    PTuneMixin = ptune_module.PTuneMixin

def teardown_module():
    patcher.stop()

class DummyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)

        # In a real environment, we would inherit from PTuneMixin directly,
        # but because it's loaded dynamically we just add it to __class__.__bases__
        self.__class__.__bases__ = (nn.Module, PTuneMixin)
        self.init_prompts(config)

def test_ptune_intermediate_embeddings():
    config = PretrainedConfig()
    config.hidden_size = 128
    config.num_hidden_layers = 4
    config.pre_seq_len = 5
    config.tuning_mode = "deep_ptune"

    model = DummyModel(config)

    # 1. Verify weights shape is (pre_seq_len, (num_layers - 1) * hidden_size)
    assert model.intermediate_prompt_embeddings.weight.shape == (5, 3 * 128)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    # 2. Verify output shapes
    assert prompts.shape == (2, 5, 128)
    assert intermediate_prompts.shape == (4, 2, 5, 128)

    # 3. Verify first layer prompts are all zeros padding
    assert torch.all(intermediate_prompts[0] == 0)

if __name__ == "__main__":
    setup_module()
    test_ptune_intermediate_embeddings()
    teardown_module()
