import os
import sys
from unittest import mock
import importlib.util

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_mixin_deep_ptune():
    """
    Test PTuneMixin with deep_ptune to ensure intermediate_prompt_embeddings
    have the correct shape of (num_hidden_layers - 1) * hidden_size.
    """
    # Use importlib to load ptune directly bypassing the parent init
    # which requires hivemind and other missing dependencies
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)

    # Mock hivemind before executing module
    sys.modules['hivemind'] = mock.MagicMock()

    # Mock petals.utils.misc.DUMMY
    petals_utils_misc = mock.MagicMock()
    petals_utils_misc.DUMMY = torch.empty(0)
    sys.modules['petals.utils.misc'] = petals_utils_misc
    sys.modules['petals.utils'] = mock.MagicMock()
    sys.modules['petals'] = mock.MagicMock()

    spec.loader.exec_module(ptune)
    PTuneMixin = ptune.PTuneMixin

    class DummyModel(PTuneMixin, nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.init_prompts(config)
            self.word_embeddings = nn.Embedding(10, config.hidden_size)

    # Test parameters
    batch_size = 2
    pre_seq_len = 5
    hidden_size = 16
    num_hidden_layers = 4

    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=pre_seq_len,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers
    )

    model = DummyModel(config)

    # Verify intermediate embedding weights size
    expected_weight_shape = (pre_seq_len, (num_hidden_layers - 1) * hidden_size)
    assert model.intermediate_prompt_embeddings.weight.shape == expected_weight_shape

    # Verify output from get_prompt
    prompts, intermediate = model.get_prompt(batch_size=batch_size)

    expected_prompts_shape = (batch_size, pre_seq_len, hidden_size)
    assert prompts.shape == expected_prompts_shape

    expected_intermediate_shape = (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate.shape == expected_intermediate_shape
