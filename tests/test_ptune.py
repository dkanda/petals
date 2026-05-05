import torch
import torch.nn as nn
from unittest import mock
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shape():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        num_hidden_layers=3,
        hidden_size=8
    )
    with mock.patch('petals.client.ptune._original_register_parameter', nn.Module.register_parameter):
        model = DummyModel(config)

    prompts, intermediate = model.get_prompt(batch_size=2)
    assert intermediate.shape == (2, 2, 5, 8)
