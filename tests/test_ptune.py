import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin, PTuneConfig
from petals.utils.misc import DUMMY

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=3
    )
    model = DummyModel(config)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts is DUMMY

def test_deep_ptune_shapes():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=3
    )
    model = DummyModel(config)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    # Check that intermediate prompt embeddings has correct number of parameters
    assert model.intermediate_prompt_embeddings.weight.shape == (5, 2 * 16)

    # Check prompts shape
    assert prompts.shape == (2, 5, 16)

    # Check intermediate_prompts shape - should have length config.num_hidden_layers due to padding
    assert intermediate_prompts.shape == (3, 2, 5, 16)

    # Check that the first element of intermediate_prompts along dim 0 is zero-padded
    assert torch.all(intermediate_prompts[0] == 0)

    # Check that the remaining elements are not exactly zero (unless weights randomly initialized to 0, highly unlikely)
    assert not torch.all(intermediate_prompts[1:] == 0)
