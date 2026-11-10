import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    import hivemind
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("hivemind not available", allow_module_level=True)


class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        pre_seq_len=8,
        tuning_mode="deep_ptune"
    )

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == torch.Size([2, 8, 16])
    assert intermediate_prompts.shape == torch.Size([3, 2, 8, 16])
