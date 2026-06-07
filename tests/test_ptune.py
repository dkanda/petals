import sys
import os
import torch
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_shapes():
    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_mock.utils = petals_utils_mock
    petals_utils_misc_mock = mock.MagicMock()
    petals_utils_misc_mock.DUMMY = torch.empty(0)
    petals_utils_mock.misc = petals_utils_misc_mock

    with mock.patch.dict('sys.modules', {
        'petals': petals_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
        'hivemind': mock.MagicMock(),
        'petals.client.inference_session': mock.MagicMock(),
        'petals.client.remote_sequential': mock.MagicMock(),
        'petals.client.routing': mock.MagicMock(),
    }):
        from petals.client.ptune import PTuneMixin
        from transformers import PretrainedConfig

        class DummyModel(PTuneMixin):
            def __init__(self):
                self.config = PretrainedConfig()
                self.config.hidden_size = 64
                self.config.num_hidden_layers = 4
                self.config.pre_seq_len = 10
                self.config.tuning_mode = "deep_ptune"
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight = torch.zeros((1,), dtype=torch.float32, device='cpu')

        model = DummyModel()
        model.init_prompts(model.config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 10, 64)
        assert intermediate_prompts.shape == (3, 2, 10, 64)
