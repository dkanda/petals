import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Skipping PTune test due to missing dependencies (e.g. hivemind)", allow_module_level=True)

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=5
    )
    model = MockModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == torch.Size([batch_size, config.pre_seq_len, config.hidden_size])
    assert intermediate_prompts.shape == torch.Size([config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size])
