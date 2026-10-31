import pytest
import torch
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
    from petals.client.ptune import force_non_empty_weights
except ImportError:
    pytest.skip("Missing dependencies (e.g., hivemind) in local environment", allow_module_level=True)

class MockConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pre_seq_len = 10
        self.hidden_size = 32
        self.num_hidden_layers = 4
        self.tuning_mode = "deep_ptune"

class MockWordEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.empty((1,), dtype=torch.float32)

class MockModel(PTuneMixin, torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pre_seq_len = config.pre_seq_len
        self.word_embeddings = MockWordEmbeddings()
        self.init_prompts(config)

def test_deep_ptune_intermediate_prompts_shape():
    config = MockConfig()
    model = MockModel(config)

    # Test shape of embeddings
    assert model.intermediate_prompt_embeddings.weight.shape == (10, 96)

    # Test shapes of get_prompt outputs
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 10, 32)
    assert intermediate_prompts.shape == (3, 2, 10, 32)
