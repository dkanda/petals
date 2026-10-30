import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Skipping test due to missing dependencies like hivemind", allow_module_level=True)

class MockModel(nn.Module, PTuneMixin):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=4,
            num_hidden_layers=3,
            hidden_size=8,
        )
        self.word_embeddings = nn.Embedding(10, 8)
        self.init_prompts(self.config)

def test_ptune_intermediate_prompts_shape():
    model = MockModel()
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 4, 8)
    assert intermediate_prompts.shape == (2, 2, 4, 8), \
        "Intermediate prompts shape should be [num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size]"
