import os
import sys

os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

import pytest
try:
    import torch
    import transformers
except ImportError:
    pytest.skip("Skipping because of missing module", allow_module_level=True)

sys.path.insert(0, os.path.abspath('src'))
try:
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("Skipping because of missing module (hivemind)", allow_module_level=True)

class DummyWordEmbeddings:
    def __init__(self, hidden_size):
        self.weight = torch.zeros((1, hidden_size), dtype=torch.float32)

class MockModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = DummyWordEmbeddings(config.hidden_size)
        self.init_prompts(config)

def test_ptune_mixin_intermediate_prompts_shape():
    config = transformers.PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=5,
        tuning_mode="deep_ptune",
        pre_seq_len=8
    )

    model = MockModel(config)
    batch_size = 3
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # Check intermediate prompts shape: (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    expected_shape = (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == expected_shape, f"Expected {expected_shape}, got {intermediate_prompts.shape}"
