import torch
import torch.nn as nn
from transformers import PretrainedConfig
from unittest import mock

import petals.client.ptune
from petals.client.ptune import PTuneMixin

class MockPTuneModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4
    )

    with mock.patch.object(petals.client.ptune, '_original_register_parameter', nn.Module.register_parameter):
        model = MockPTuneModel(config)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (3, 2, 5, 16)
