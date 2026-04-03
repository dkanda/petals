import sys
import unittest.mock as mock

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

# Apply module mocks *before* importing anything that depends on them
sys.modules['hivemind'] = MockPackage()
sys.modules['tensor_parallel'] = MockPackage()
sys.modules['petals.utils.misc'] = mock.MagicMock()

import torch
sys.modules['petals.utils.misc'].DUMMY = torch.tensor([0])
sys.modules['petals'] = MockPackage()

import pytest
import torch.nn as nn
from transformers import PretrainedConfig

import importlib.util
spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
ptune = importlib.util.module_from_spec(spec)
sys.modules["ptune"] = ptune
spec.loader.exec_module(ptune)

class MockModel(nn.Module, ptune.PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_intermediate_prompt_shape():
    """
    Test that deep_ptune correctly creates intermediate prompts with
    shape [num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size]
    """
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )

    model = MockModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    # Test main prompt shape
    assert prompts.shape == torch.Size([batch_size, config.pre_seq_len, config.hidden_size]), \
        f"Expected prompts shape {[batch_size, config.pre_seq_len, config.hidden_size]}, got {prompts.shape}"

    # Test intermediate prompt shape which should be num_hidden_layers - 1
    expected_intermediate_shape = torch.Size([
        config.num_hidden_layers - 1,
        batch_size,
        config.pre_seq_len,
        config.hidden_size
    ])

    assert intermediate_prompts.shape == expected_intermediate_shape, \
        f"Expected intermediate prompts shape {expected_intermediate_shape}, got {intermediate_prompts.shape}"
