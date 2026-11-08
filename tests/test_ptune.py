import pytest
import torch
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Missing dependencies (e.g., hivemind) in local test environment.", allow_module_level=True)

class MockModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompt_shape():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=32,
        num_hidden_layers=4
    )
    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == torch.Size([2, 5, 32])
    # For deep_ptune, the expected shape is (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == torch.Size([3, 2, 5, 32])

def test_ptune_basic_prompt_shape():
    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=5,
        hidden_size=32,
        num_hidden_layers=4
    )
    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == torch.Size([2, 5, 32])
    assert intermediate_prompts.shape == torch.Size([0]) # DUMMY tensor shape
