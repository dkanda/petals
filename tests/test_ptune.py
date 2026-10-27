import os
import pytest
import torch
import sys

sys.path.insert(0, os.path.abspath('src'))

# Only run if dependencies are available, otherwise skip
try:
    from petals.client.ptune import PTuneMixin, PTuneConfig
    from transformers import PretrainedConfig
except ImportError:
    pytest.skip("Skipping tests because missing optional dependencies", allow_module_level=True)

class MockModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = torch.nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=10,
        hidden_size=32,
        num_hidden_layers=4,
    )

    # We use a context manager to patch `force_non_empty_weights` behavior if needed,
    # but the MockModel will just initialize `nn.Embedding` as usual in this test context.
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    # The intermediate prompts should be sized for num_hidden_layers - 1
    # with shape: (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)

if __name__ == "__main__":
    test_ptune_intermediate_prompts_shape()
    print("Test passed!")
