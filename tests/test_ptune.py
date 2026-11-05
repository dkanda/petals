import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Skipping PTune test due to missing deep dependencies (e.g. hivemind)", allow_module_level=True)


class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)


def test_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=3,
        tuning_mode="deep_ptune",
        pre_seq_len=4
    )

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 4, 16)
    assert intermediate_prompts.shape == (2, 2, 4, 16)
