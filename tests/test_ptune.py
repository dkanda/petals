import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin, PTuneConfig
from petals.utils.misc import DUMMY

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_deep_ptune():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=4,
    )
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 8

    with patch("petals.client.ptune._original_register_parameter", nn.Module.register_parameter):
        model = DummyModel(config)

    assert model.pre_seq_len == 8
    assert model.prompt_embeddings.weight.shape == (8, 64)
    assert model.intermediate_prompt_embeddings.weight.shape == (8, 3 * 64)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 8, 64)
    assert intermediate_prompts.shape == (3, 2, 8, 64)

def test_ptune_ptune_only():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=4,
    )
    config.tuning_mode = "ptune"
    config.pre_seq_len = 8

    with patch("petals.client.ptune._original_register_parameter", nn.Module.register_parameter):
        model = DummyModel(config)

    assert model.pre_seq_len == 8
    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 8, 64)

    assert intermediate_prompts is DUMMY or "DUMMY.to()" in repr(intermediate_prompts) or (isinstance(intermediate_prompts, torch.Tensor) and intermediate_prompts.numel() == 0)
