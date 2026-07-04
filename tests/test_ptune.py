import os
import sys
sys.path.insert(0, os.path.abspath('src'))

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
    from petals.utils.misc import DUMMY
except ImportError:
    pytest.skip("Could not import petals", allow_module_level=True)


class DummyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pre_seq_len = 10
        self.hidden_size = 32
        self.num_hidden_layers = 12


class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)


def test_ptune_mixin_intermediate_prompts_shape():
    config = DummyConfig(tuning_mode="deep_ptune")
    model = DummyModel(config)

    # Verify the weight shape
    expected_weight_shape = (config.pre_seq_len, (config.num_hidden_layers - 1) * config.hidden_size)
    assert model.intermediate_prompt_embeddings.weight.shape == torch.Size(expected_weight_shape)

    # Verify the output prompts shape
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    expected_prompts_shape = (batch_size, config.pre_seq_len, config.hidden_size)
    assert prompts.shape == torch.Size(expected_prompts_shape)

    expected_intermediate_shape = (
        config.num_hidden_layers - 1,
        batch_size,
        config.pre_seq_len,
        config.hidden_size
    )
    assert intermediate_prompts.shape == torch.Size(expected_intermediate_shape)


def test_ptune_mixin_normal_tuning_mode():
    config = DummyConfig(tuning_mode="ptune")
    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    expected_prompts_shape = (batch_size, config.pre_seq_len, config.hidden_size)
    assert prompts.shape == torch.Size(expected_prompts_shape)

    # In normal ptune mode, intermediate prompts should be DUMMY
    assert intermediate_prompts is DUMMY
