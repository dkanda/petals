import sys
import os
import torch
import pytest
from transformers import PretrainedConfig

class MockConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 64
        self.num_hidden_layers = 12
        self.pre_seq_len = 16
        self.tuning_mode = "deep_ptune"

def test_ptune_mixin():
    try:
        from petals.client.ptune import PTuneMixin
    except ImportError:
        pytest.skip("Cannot import petals locally due to missing hivemind dependencies")

    class MockWordEmbeddings:
        def __init__(self):
            self.weight = torch.empty(0, dtype=torch.float32)

    class MockModel(PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = MockWordEmbeddings()
            self.init_prompts(config)

    config = MockConfig()
    model = MockModel(config)

    prompts, intermediate_prompts = model.get_prompt(batch_size=4)
    assert prompts.shape == torch.Size([4, 16, 64])
    assert intermediate_prompts.shape == torch.Size([11, 4, 16, 64]), f"Got {intermediate_prompts.shape}"

if __name__ == '__main__':
    test_ptune_mixin()
    print("Passed.")
