import torch
import torch.nn as nn
from petals.client.ptune import PTuneMixin

class DummyConfig:
    pre_seq_len = 5
    tuning_mode = "deep_ptune"
    hidden_size = 16
    num_hidden_layers = 4

class DummyModel(PTuneMixin, torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = torch.nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_deep_ptune():
    config = DummyConfig()
    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # Check first layer is zero padding
    assert torch.all(intermediate_prompts[0] == 0)
