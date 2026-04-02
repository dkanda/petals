import torch
import torch.nn as nn
from transformers import PretrainedConfig
import pytest

from petals.client.ptune import PTuneMixin

class DummyWordEmbeddings:
    def __init__(self):
        self.weight = torch.empty((1,), dtype=torch.float32, device="cpu")

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = DummyWordEmbeddings()
        self.init_prompts(config)

def test_deep_ptune_intermediate_prompts_shape():
    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5
    config.hidden_size = 16
    config.num_hidden_layers = 4

    model = DummyModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
