import sys
import os

def test_ptune_layer_count():
    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig
    from unittest import mock

    # Setup mocks
    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_utils_misc_mock = mock.MagicMock(__spec__=None)

    petals_mock.client = petals_client_mock
    petals_mock.utils = petals_utils_mock
    petals_utils_mock.misc = petals_utils_misc_mock

    petals_utils_misc_mock.DUMMY = torch.empty(0)

    hivemind_mock = mock.MagicMock(__path__=[], __spec__=None)

    mocks = {
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
        'hivemind': hivemind_mock,
    }

    with mock.patch.dict('sys.modules', mocks):
        import petals.client.ptune as ptune

        # Patch the locally captured register_parameter
        with mock.patch.object(ptune, '_original_register_parameter', nn.Module.register_parameter):
            class DummyConfig(PretrainedConfig):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.tuning_mode = "deep_ptune"
                    self.pre_seq_len = 5
                    self.hidden_size = 8
                    self.num_hidden_layers = 4

            class DummyModel(ptune.PTuneMixin, nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            config = DummyConfig()
            model = DummyModel(config)

            assert model.intermediate_prompt_embeddings.weight.shape == torch.Size([5, 24])

            _, intermediate_prompts = model.get_prompt(batch_size=2)
            assert intermediate_prompts.shape == torch.Size([3, 2, 5, 8])
