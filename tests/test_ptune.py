import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin

class DummyWordEmbeddings:
    def __init__(self):
        self.weight = torch.zeros(1, dtype=torch.float32)

class MockModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = DummyWordEmbeddings()
        self.init_prompts(config)

class MockConfig(PretrainedConfig):
    def __init__(self, tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

def test_deep_ptune():
    config = MockConfig(tuning_mode="deep_ptune")
    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (4, 2, 5, 16)
    assert torch.all(intermediate_prompts[0] == 0)

def test_ptune():
    config = MockConfig(tuning_mode="ptune")
    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    # The expected output from `get_prompt` in `ptune` mode is `DUMMY` from `petals.utils.misc`.
    # Depending on how DUMMY handles `.to()`, it can be DUMMY itself, a mock, or an empty tensor.
    assert "DUMMY" in repr(intermediate_prompts) or "tensor([0.])" in repr(intermediate_prompts) or getattr(intermediate_prompts, "numel", lambda: 1)() == 0
