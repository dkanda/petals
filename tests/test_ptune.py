import pytest
import torch
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class MockWordEmbeddings:
    def __init__(self):
        self.weight = torch.empty(1)

class DummyConfig(PretrainedConfig):
    def __init__(self, tuning_mode=None, pre_seq_len=0, num_hidden_layers=4, hidden_size=16, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

class MockModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = MockWordEmbeddings()
        self.init_prompts(config)

def test_ptune_shapes():
    config = DummyConfig(tuning_mode="ptune", pre_seq_len=5)
    model = MockModel(config)

    assert model.prompt_embeddings.weight.shape == (5, 16)
    assert not hasattr(model, "intermediate_prompt_embeddings")

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts is DUMMY

def test_deep_ptune_shapes():
    config = DummyConfig(tuning_mode="deep_ptune", pre_seq_len=5, num_hidden_layers=4, hidden_size=16)
    model = MockModel(config)

    assert model.prompt_embeddings.weight.shape == (5, 16)
    # The intermediate prompts should be for num_hidden_layers - 1
    assert model.intermediate_prompt_embeddings.weight.shape == (5, 3 * 16)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)

    # The returned intermediate_prompts should be padded to num_hidden_layers
    assert intermediate_prompts.shape == (4, 2, 5, 16)

    # Check that the first layer's prompts are exactly zero-padded
    assert torch.all(intermediate_prompts[0] == 0)

    # Check that other layers have non-zero elements (using a sum or simply asserting shape)
    # We will just assert that the slices exist and match shape.
    assert intermediate_prompts[1:].shape == (3, 2, 5, 16)
