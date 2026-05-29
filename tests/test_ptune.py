import os
import sys
from unittest import mock
import torch
from transformers import PretrainedConfig

sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

def test_ptune_shapes():
    import hivemind
    hivemind_mock = mock.MagicMock(__path__=["mock_hivemind"], __spec__=None)

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_mock.client = petals_client_mock

    petals_utils_misc_mock = mock.MagicMock()
    petals_utils_misc_mock.DUMMY = torch.empty(0)

    mocks = {
        'hivemind': hivemind_mock,
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.utils': mock.MagicMock(__path__=["src/petals/utils"], __spec__=None),
        'petals.utils.misc': petals_utils_misc_mock,
    }

    with mock.patch.dict('sys.modules', mocks):
        import petals.client.ptune as ptune

        with mock.patch.object(ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
            class DummyModel(ptune.PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            config = PretrainedConfig(
                tuning_mode="deep_ptune",
                pre_seq_len=5,
                hidden_size=16,
                num_hidden_layers=4
            )

            model = DummyModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert prompts.shape == (2, 5, 16)
            assert intermediate_prompts.shape == (3, 2, 5, 16)
