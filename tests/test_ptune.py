import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    import hivemind
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("hivemind not available", allow_module_level=True)

class MockModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5
    config.hidden_size = 16
    config.num_hidden_layers = 4

    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 16)
    # Expected shape: (batch_size, pre_seq_len, num_hidden_layers - 1, hidden_size)
    # Actually it permutes: permute([2, 0, 1, 3])
    # Let's verify the shape
    # intermediate_prompts.view(batch_size, pre_seq_len, num_hidden_layers - 1, hidden_size)
    # After permute([2, 0, 1, 3]):
    # Dim 0 -> 2 (num_hidden_layers - 1)
    # Dim 1 -> 0 (batch_size)
    # Dim 2 -> 1 (pre_seq_len)
    # Dim 3 -> 3 (hidden_size)
    # So expected shape: (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (3, 2, 5, 16)

if __name__ == "__main__":
    pytest.main([__file__])