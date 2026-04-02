import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin

class MockWordEmbeddings:
    def __init__(self):
        self.weight = nn.Parameter(torch.empty(0, dtype=torch.float32, device=torch.device('cpu')))

class MockModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = MockWordEmbeddings()
        self.init_prompts(config)

def test_deep_ptune():
    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5
    config.hidden_size = 16
    config.num_hidden_layers = 4

    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check prompts shape
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert prompts.dtype == torch.float32

    # Check intermediate_prompts shape
    assert intermediate_prompts.shape == (
        config.num_hidden_layers,
        batch_size,
        config.pre_seq_len,
        config.hidden_size,
    )
    assert intermediate_prompts.dtype == torch.float32

    # Check padding of first layer
    # The first layer should be padded with zeros
    first_layer_prompts = intermediate_prompts[0, :, :, :]
    assert torch.all(first_layer_prompts == 0)

    # Check that subsequent layers are not zeroed
    subsequent_layer_prompts = intermediate_prompts[1:, :, :, :]
    # Ensure they have some non-zero values (since they are initialized as nn.Embedding weights)
    assert torch.any(subsequent_layer_prompts != 0)
