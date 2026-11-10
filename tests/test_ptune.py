import sys
import os
import pytest
sys.path.insert(0, os.path.abspath('src'))
os.environ["PETALS_IGNORE_DEPENDENCY_VERSION"] = "1"

try:
    import petals.client.ptune
    from transformers import PretrainedConfig
    import torch
    import torch.nn as nn
    from petals.client.ptune import PTuneMixin
except ImportError as e:
    pytest.skip(f"Skipping test due to missing dependencies: {e}", allow_module_level=True)

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_shapes():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=64,
        num_hidden_layers=3
    )
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 64)
    assert intermediate_prompts.shape == (2, 2, 5, 64)
