import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin, PTuneConfig
from petals.utils.misc import DUMMY

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_mixin_ptune_mode():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="ptune",
        pre_seq_len=5,
    )
    model = DummyModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")
    assert model.prompt_embeddings.weight.shape == (5, 16)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)
    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts is DUMMY

def test_ptune_mixin_deep_ptune_mode():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=5,
    )
    model = DummyModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")
    assert model.prompt_embeddings.weight.shape == (5, 16)
    # parameter allocation for num_hidden_layers - 1
    assert model.intermediate_prompt_embeddings.weight.shape == (5, (4 - 1) * 16)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)
    assert prompts.shape == (2, 5, 16)
    # output shape needs to contain zero padding so size is 4, bat=2, seq=5, hid=16
    assert intermediate_prompts.shape == (4, 2, 5, 16)

    # Verify the first layer is padded with zeros
    assert torch.all(intermediate_prompts[0] == 0)
