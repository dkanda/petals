import torch
import torch.nn as nn
from transformers import PretrainedConfig
import unittest.mock
from petals.client.ptune import PTuneMixin

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_deep_ptune():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )
    with unittest.mock.patch('petals.client.ptune._original_register_parameter', torch.nn.Module.register_parameter):
        model = DummyModel(config)

        # intermediate_prompt_embeddings weight shape should be (pre_seq_len, (num_hidden_layers - 1) * hidden_size)
        assert model.intermediate_prompt_embeddings.weight.shape == (5, 3 * 16)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert prompts.shape == (2, 5, 16)
        # intermediate_prompts shape should be (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
        assert intermediate_prompts.shape == (3, 2, 5, 16)
