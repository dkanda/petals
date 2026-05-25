import sys
import os
import torch
import torch.nn as nn
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_shapes():
    import hivemind
    hivemind.PeerID = hivemind.p2p.PeerID
    hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
    hivemind.get_logger = hivemind.utils.get_logger

    petals_mock = mock.MagicMock()
    petals_mock.__path__ = ["src/petals"]
    petals_mock.__spec__ = None

    petals_client_mock = mock.MagicMock()
    petals_client_mock.__path__ = ["src/petals/client"]
    petals_client_mock.__spec__ = None

    petals_mock.client = petals_client_mock

    petals_utils_mock = mock.MagicMock()
    petals_utils_mock.__path__ = ["src/petals/utils"]
    petals_utils_mock.__spec__ = None
    petals_mock.utils = petals_utils_mock

    petals_utils_misc_mock = mock.MagicMock()
    petals_utils_misc_mock.DUMMY = torch.empty(0)
    petals_utils_mock.misc = petals_utils_misc_mock

    with mock.patch.dict('sys.modules', {
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
        'petals.client.inference_session': mock.MagicMock(),
        'petals.client.remote_sequential': mock.MagicMock(),
        'petals.client.routing': mock.MagicMock(),
    }):
        from petals.client.ptune import PTuneMixin, PTuneConfig
        from transformers import PretrainedConfig
        import petals.client.ptune

        class MockConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.hidden_size = 64
                self.num_hidden_layers = 12
                self.pre_seq_len = 8
                self.tuning_mode = "deep_ptune"

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)

                # Mock _original_register_parameter because torch 2.x breaks deep down in empty weight init contexts
                with mock.patch.object(petals.client.ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
                    self.init_prompts(config)

        config = MockConfig()
        model = DummyModel(config)

        # Test shape
        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        assert prompts.shape == torch.Size([2, 8, 64])
        assert intermediate_prompts.shape == torch.Size([11, 2, 8, 64])
