import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    class DummyConfig(PretrainedConfig):
        def __init__(self):
            super().__init__()
            self.tuning_mode = "ptune"
            self.pre_seq_len = 5
            self.hidden_size = 16
            self.num_hidden_layers = 4

    config = DummyConfig()

    model = DummyModel(config)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    assert ("DUMMY.to()" in repr(intermediate_prompts) or
            intermediate_prompts is DUMMY or
            (isinstance(intermediate_prompts, torch.Tensor) and intermediate_prompts.numel() == 0))

def test_deep_ptune_shapes():
    class DummyConfig(PretrainedConfig):
        def __init__(self):
            super().__init__()
            self.tuning_mode = "deep_ptune"
            self.pre_seq_len = 5
            self.hidden_size = 16
            self.num_hidden_layers = 4

    config = DummyConfig()

    model = DummyModel(config)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    # expected shape: (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (3, 2, 5, 16)
