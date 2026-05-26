import sys
import os
sys.path.insert(0, os.path.abspath('src'))

from unittest import mock
import torch

def test_ptune_mixin_deep_ptune():
    import hivemind
    hivemind.PeerID = hivemind.p2p.PeerID
    hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
    hivemind.get_logger = hivemind.utils.get_logger

    os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

    mocks = {
        'petals.client.inference_session': mock.MagicMock(__path__=[], __spec__=None),
        'petals.client.remote_sequential': mock.MagicMock(__path__=[], __spec__=None),
        'petals.client.routing': mock.MagicMock(__path__=[], __spec__=None),
    }
    with mock.patch.dict('sys.modules', mocks):
        with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
            from petals.client.ptune import PTuneMixin

            class DummyModel(PTuneMixin):
                def __init__(self):
                    self.word_embeddings = mock.MagicMock()
                    self.word_embeddings.weight.device = torch.device('cpu')
                    self.word_embeddings.weight.dtype = torch.float32
                    class Config:
                        tuning_mode = "deep_ptune"
                        pre_seq_len = 5
                        hidden_size = 16
                        num_hidden_layers = 4
                    self.config = Config()

            model = DummyModel()
            with mock.patch.object(sys.modules['petals.client.ptune'], '_original_register_parameter', torch.nn.Module.register_parameter):
                model.init_prompts(model.config)

            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert prompts.shape == (2, 5, 16)
            assert intermediate_prompts.shape == (3, 2, 5, 16)
