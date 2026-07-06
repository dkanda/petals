import pytest
import torch

try:
    from transformers import PretrainedConfig
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Skipping test due to missing dependencies", allow_module_level=True)

class DummyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 64
        self.num_hidden_layers = 12
        self.pre_seq_len = 5
        self.tuning_mode = "deep_ptune"

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = torch.nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = DummyConfig()
    model = DummyModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
