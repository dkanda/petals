import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Skipping test due to missing dependencies like hivemind", allow_module_level=True)

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_intermediate_prompt_embeddings_shape():
    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5
    config.hidden_size = 8
    config.num_hidden_layers = 4

    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == torch.Size([2, 5, 8])
    assert intermediate_prompts.shape == torch.Size([3, 2, 5, 8]), \
        f"Expected intermediate prompts shape (3, 2, 5, 8), got {intermediate_prompts.shape}"
