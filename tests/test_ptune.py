import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin

class MockConfig(PretrainedConfig):
    def __init__(self, tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=8, num_hidden_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_mixin_deep_ptune():
    config = MockConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=8, num_hidden_layers=4)
    model = MockModel(config)

    # Assert intermediate prompts have correct number of parameters
    # Should be (num_hidden_layers - 1) * hidden_size
    assert model.intermediate_prompt_embeddings.weight.shape == (5, 3 * 8)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Assert prompts shape: (batch_size, pre_seq_len, hidden_size)
    assert prompts.shape == (2, 5, 8)

    # Assert intermediate_prompts shape
    # The return shape after permutation and padding should be:
    # (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (4, 2, 5, 8)

    # Assert that the first layer (index 0) is all zeros
    assert torch.all(intermediate_prompts[0] == 0)

def test_ptune_mixin_ptune():
    config = MockConfig(tuning_mode="ptune", pre_seq_len=5, hidden_size=8, num_hidden_layers=4)
    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (2, 5, 8)
    # in standard ptune, intermediate_prompts should be DUMMY tensor (which is empty/dummy)
    # DUMMY is an instance of petals.utils.misc.DUMMY
    from petals.utils.misc import DUMMY
    assert intermediate_prompts is DUMMY or (hasattr(intermediate_prompts, "numel") and intermediate_prompts.numel() == 0)
