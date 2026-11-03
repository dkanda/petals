import pytest
import torch
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Skipping test due to missing dependencies like hivemind", allow_module_level=True)

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=5,
        tuning_mode="deep_ptune",
        pre_seq_len=4
    )
    model = DummyModel(config)
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    # Assert intermediate prompts shape
    expected_shape = (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == expected_shape, f"Expected {expected_shape}, got {intermediate_prompts.shape}"
