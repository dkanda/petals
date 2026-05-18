import sys
from unittest import mock
import torch
import torch.nn as nn
import os

def test_ptune_shapes():
    import hivemind

    petals_server_mock = mock.MagicMock()
    petals_server_mock.__path__ = []
    petals_server_mock.__spec__ = None
    petals_server_handler_mock = mock.MagicMock()
    petals_server_mock.handler = petals_server_handler_mock
    petals_models_mock = mock.MagicMock()
    petals_models_mock.__path__ = []
    petals_models_mock.__spec__ = None

    os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

    with mock.patch.dict('sys.modules', {
        'hivemind': mock.MagicMock(),
        'petals.server': petals_server_mock,
        'petals.server.handler': petals_server_handler_mock,
        'petals.models': petals_models_mock
    }):
        from petals.client.ptune import PTuneMixin
        import petals.client.ptune as ptune_module

        with mock.patch.object(ptune_module, '_original_register_parameter', nn.Module.register_parameter):
            class MockModel(PTuneMixin, nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            class Config:
                tuning_mode = "deep_ptune"
                pre_seq_len = 5
                hidden_size = 8
                num_hidden_layers = 3

            config = Config()
            model = MockModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert prompts.shape == (2, 5, 8)
            assert intermediate_prompts.shape == (2, 2, 5, 8)
