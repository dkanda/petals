import os
import sys

def test_ptune_mixin():
    from unittest import mock
    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)

    hivemind_mock = mock.MagicMock()
    hivemind_mock.get_logger = mock.MagicMock()

    petals_utils_misc_mock = mock.MagicMock()
    petals_utils_misc_mock.DUMMY = torch.empty(0)

    with mock.patch.dict('sys.modules', {
        'petals.client': petals_client_mock,
        'hivemind': hivemind_mock,
        'petals.utils.misc': petals_utils_misc_mock
    }):
        from petals.client.ptune import PTuneMixin

        class MockModel(PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            hidden_size=16,
            num_hidden_layers=4,
            tuning_mode="deep_ptune",
            pre_seq_len=5
        )

        model = MockModel(config)
        assert model.intermediate_prompt_embeddings.weight.shape == (5, (4 - 1) * 16), f"Actual shape: {model.intermediate_prompt_embeddings.weight.shape}"

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert prompts.shape == (2, 5, 16)
        assert intermediate_prompts.shape == (4 - 1, 2, 5, 16)
