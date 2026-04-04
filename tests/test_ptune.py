import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import sys
import unittest.mock as mock

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

with mock.patch.dict(sys.modules, {
    'hivemind': MockPackage(),
    'hivemind.get_logger': mock.MagicMock(),
    'petals.utils.misc': mock.MagicMock(),
}):
    # Import inside the mock context
    import importlib.util
    spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)

    # Inject DUMMY
    from petals.utils.misc import DUMMY
    ptune.DUMMY = DUMMY

    spec.loader.exec_module(ptune)
    PTuneMixin = ptune.PTuneMixin


class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="ptune",
        pre_seq_len=5
    )
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

def test_deep_ptune_shapes():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=5
    )
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    # the shape of intermediate_prompts is (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # check that the first layer (index 0) of intermediate_prompts is strictly zero
    assert torch.all(intermediate_prompts[0] == 0)

    # ensure it uses the correct precision
    assert intermediate_prompts.dtype == model.word_embeddings.weight.dtype
