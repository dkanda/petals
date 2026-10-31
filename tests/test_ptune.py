import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

class TestPTuneMixin:
    def test_ptune_intermediate_shape(self):
        try:
            import hivemind
        except ImportError:
            pytest.skip("hivemind not available", allow_module_level=True)

        from petals.client.ptune import PTuneMixin
        class MockModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=16,
            hidden_size=64,
            num_hidden_layers=12
        )
        model = MockModel(config)
        prompts, intermediate = model.get_prompt(2)

        assert prompts.shape == (2, 16, 64)
        assert intermediate.shape == (11, 2, 16, 64)
