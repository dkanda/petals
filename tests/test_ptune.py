import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin

class DummyConfig(PretrainedConfig):
    def __init__(self, tuning_mode="deep_ptune", pre_seq_len=8, hidden_size=16, num_hidden_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_intermediate_prompts():
    config = DummyConfig()
    model = DummyModel(config)

    # Assert intermediate_prompt_embeddings layer size is (num_hidden_layers - 1) * hidden_size
    assert model.intermediate_prompt_embeddings.embedding_dim == (config.num_hidden_layers - 1) * config.hidden_size

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check shape of intermediate_prompts: [num_hidden_layers, batch_size, pre_seq_len, hidden_size]
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # Check that the first layer (index 0) is all zeros
    assert torch.all(intermediate_prompts[0] == 0)

if __name__ == "__main__":
    pytest.main([__file__])
