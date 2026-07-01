import os
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'
import sys
sys.path.insert(0, os.path.abspath('src'))
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin
from unittest import mock

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        hidden_size=128,
        num_hidden_layers=10,
        tuning_mode="deep_ptune",
        pre_seq_len=5
    )

    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    # expected shape: (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (9, 2, 5, 128)
