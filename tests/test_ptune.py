import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

try:
    import hivemind
    from petals.client.ptune import PTuneMixin
    from petals.utils.misc import DUMMY
except ImportError:
    pytest.skip("hivemind not available", allow_module_level=True)

class DummyConfig(PretrainedConfig):
    def __init__(self, tuning_mode="deep_ptune", pre_seq_len=5, num_hidden_layers=10, hidden_size=64, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = DummyConfig()
    model = DummyModel(config)

    # Check weight shape
    expected_weight_shape = (config.pre_seq_len, (config.num_hidden_layers - 1) * config.hidden_size)
    assert model.intermediate_prompt_embeddings.weight.shape == expected_weight_shape, \
        f"Expected {expected_weight_shape}, got {model.intermediate_prompt_embeddings.weight.shape}"

    # Check output shapes
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    expected_prompts_shape = (batch_size, config.pre_seq_len, config.hidden_size)
    assert prompts.shape == expected_prompts_shape, \
        f"Expected {expected_prompts_shape}, got {prompts.shape}"

    expected_intermediate_shape = (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == expected_intermediate_shape, \
        f"Expected {expected_intermediate_shape}, got {intermediate_prompts.shape}"

def test_ptune_intermediate_prompts_shape_not_deep():
    config = DummyConfig(tuning_mode="ptune")
    model = DummyModel(config)

    assert not hasattr(model, "intermediate_prompt_embeddings")

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    expected_prompts_shape = (batch_size, config.pre_seq_len, config.hidden_size)
    assert prompts.shape == expected_prompts_shape

    assert intermediate_prompts is DUMMY
