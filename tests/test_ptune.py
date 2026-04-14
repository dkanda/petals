import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin

class MockConfig(PretrainedConfig):
    def __init__(self, tuning_mode="deep_ptune", pre_seq_len=5, num_hidden_layers=3, hidden_size=8, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

class MockModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_shapes():
    config = MockConfig()
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Prompt shape should be (batch_size, pre_seq_len, hidden_size)
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # Intermediate prompts shape should be (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # The first layer of intermediate prompts should be all zeros
    assert torch.all(intermediate_prompts[0] == 0)

if __name__ == "__main__":
    test_deep_ptune_shapes()
    print("All tests passed!")
