import torch
import torch.nn as nn
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class DummyConfig(PretrainedConfig):
    def __init__(self, hidden_size=64, num_hidden_layers=4, tuning_mode=None, pre_seq_len=0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    config = DummyConfig(tuning_mode="ptune", pre_seq_len=5)
    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts is DUMMY
    assert prompts.dtype == torch.float32

def test_deep_ptune_shapes():
    config = DummyConfig(tuning_mode="deep_ptune", pre_seq_len=5, num_hidden_layers=4, hidden_size=64)
    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # intermediate_prompts should have shape (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # Check that the first layer is padded with zeros
    first_layer_prompts = intermediate_prompts[0, :, :, :]
    assert torch.all(first_layer_prompts == 0)

    # Check that the remaining layers are not all zeros (which would mean they're not from the embeddings)
    remaining_layers = intermediate_prompts[1:, :, :, :]
    assert not torch.all(remaining_layers == 0)

def test_ptune_invalid_mode():
    config = DummyConfig(tuning_mode="invalid_mode", pre_seq_len=5)

    try:
        model = DummyModel(config)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        pass
