import os
import sys
import pytest

sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

try:
    from petals.client.ptune import PTuneMixin
    from transformers import PretrainedConfig
    import torch
except ImportError:
    pytest.skip("skipping due to missing dependencies", allow_module_level=True)

class DummyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pre_seq_len = 5
        self.hidden_size = 16
        self.num_hidden_layers = 4
        self.tuning_mode = "deep_ptune"

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = type('obj', (object,), {'weight': torch.zeros(1, dtype=torch.float32)})
        self.init_prompts(config)

def test_ptune():
    config = DummyConfig()
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(2)
    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (3, 2, 5, 16)
