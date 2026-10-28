import pytest
import torch
from transformers import PretrainedConfig
import sys
import os

sys.path.insert(0, os.path.abspath('src'))

try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Skipping test because petals imports fail (e.g. hivemind is missing or has missing dependencies like PeerID)", allow_module_level=True)

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = torch.nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_mixin_deep_ptune():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=5
    )

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (3, 2, 5, 16)

def test_ptune_mixin_ptune():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        tuning_mode="ptune",
        pre_seq_len=5
    )

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 16)
    from petals.utils.misc import DUMMY
    assert intermediate_prompts == DUMMY
