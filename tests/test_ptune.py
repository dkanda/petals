import pytest
import torch

try:
    from petals.client.ptune import PTuneMixin
    from transformers import PretrainedConfig
except ImportError as e:
    pytest.skip(f"Could not import petals: {e}", allow_module_level=True)

def test_ptune_mixin_intermediate_prompts_shape():
    class DummyConfig:
        tuning_mode = "deep_ptune"
        pre_seq_len = 10
        hidden_size = 32
        num_hidden_layers = 4

    class DummyWordEmbeddings:
        weight = torch.empty(0, dtype=torch.float32)

    class DummyModel(PTuneMixin):
        def __init__(self):
            self.word_embeddings = DummyWordEmbeddings()
            self.config = DummyConfig()

    model = DummyModel()
    config = DummyConfig()

    model.init_prompts(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
