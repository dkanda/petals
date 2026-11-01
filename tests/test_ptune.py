import pytest
import torch
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("skipping due to missing dependencies", allow_module_level=True)

class MockModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = type('obj', (object,), {'weight': type('obj', (object,), {'device': 'cpu', 'dtype': torch.float32})})()
        self.init_prompts(config)

def test_deep_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=10,
        hidden_size=16,
        num_hidden_layers=4
    )
    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 10, 16)
    assert intermediate_prompts.shape == (3, 2, 10, 16)
