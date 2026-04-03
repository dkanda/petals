import importlib.util
import sys
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from unittest import mock

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def load_ptune_module():
    # Isolate module testing to avoid missing dependencies like hivemind
    with mock.patch.dict(sys.modules, {'hivemind': MockPackage()}):
        petals_utils_misc = mock.MagicMock()
        petals_utils_misc.DUMMY = mock.MagicMock()
        sys.modules['petals.utils.misc'] = petals_utils_misc

        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune)
        return ptune

ptune = load_ptune_module()

class MockModel(nn.Module, ptune.PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=4,
        pre_seq_len=8,
        tuning_mode="deep_ptune"
    )
    model = MockModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (2, 8, 64)
    assert intermediate_prompts.shape == (4, 2, 8, 64)
    assert torch.all(intermediate_prompts[0] == 0)

def test_ptune():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=4,
        pre_seq_len=8,
        tuning_mode="ptune"
    )
    model = MockModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (2, 8, 64)
    # Check for DUMMY string equivalent in mock representation
    assert "DUMMY.to()" in repr(intermediate_prompts) or intermediate_prompts is sys.modules['petals.utils.misc'].DUMMY.to()
