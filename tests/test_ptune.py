import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class MockWordEmbeddings:
    def __init__(self, hidden_size):
        self.weight = torch.empty(10, hidden_size, dtype=torch.float32)

class MockModel(PTuneMixin):
    def __init__(self, config: PretrainedConfig):
        self.config = config
        self.word_embeddings = MockWordEmbeddings(config.hidden_size)
        self.init_prompts(config)

def test_ptune_mode():
    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=3
    )
    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check regular prompts
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # Check intermediate prompts
    assert intermediate_prompts is DUMMY

def test_deep_ptune_mode():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4
    )
    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")

    # The intermediate embeddings should hold parameters for num_hidden_layers - 1
    expected_intermediate_dim = (config.num_hidden_layers - 1) * config.hidden_size
    assert model.intermediate_prompt_embeddings.embedding_dim == expected_intermediate_dim

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check regular prompts
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # Check intermediate prompts
    # Shape should be (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    expected_shape = (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == expected_shape

    # Check that the first layer (index 0) is all zeros
    first_layer_prompts = intermediate_prompts[0]
    assert torch.all(first_layer_prompts == 0)

def test_ptune_invalid_mode():
    config = PretrainedConfig(
        tuning_mode="invalid_mode",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=3
    )
    with pytest.raises(NotImplementedError):
        MockModel(config)

def test_ptune_missing_pre_seq_len():
    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=0,
        hidden_size=16,
        num_hidden_layers=3
    )
    with pytest.raises(AssertionError):
        MockModel(config)
