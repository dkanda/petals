import torch
import torch.nn as nn
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin
import pytest

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompt_shape():
    config = PretrainedConfig(
        hidden_size=64,
        num_hidden_layers=10,
    )
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5

    model = DummyModel(config)

    # Test shape of embeddings directly
    assert model.intermediate_prompt_embeddings.weight.shape == torch.Size([5, 9 * 64])

    # Test output of get_prompt
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == torch.Size([2, 5, 64])
    assert intermediate_prompts.shape == torch.Size([9, 2, 5, 64])

if __name__ == "__main__":
    test_ptune_intermediate_prompt_shape()
    print("Test passed")
