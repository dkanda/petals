import os
import sys
from unittest import mock
import torch

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_intermediate_prompts_shape():
    hivemind_mock = mock.MagicMock()
    hivemind_mock.__path__ = []
    hivemind_mock.__spec__ = None
    hivemind_mock.p2p.PeerID = mock.MagicMock()
    hivemind_mock.PeerID = hivemind_mock.p2p.PeerID
    hivemind_mock.utils.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.MSGPackSerializer = hivemind_mock.utils.MSGPackSerializer
    hivemind_mock.utils.get_logger = mock.MagicMock()
    hivemind_mock.get_logger = hivemind_mock.utils.get_logger

    hivemind_mock.dht = mock.MagicMock()
    hivemind_mock.moe = mock.MagicMock()
    hivemind_mock.p2p = mock.MagicMock()

    tensor_parallel_mock = mock.MagicMock()

    petals_utils_misc_mock = mock.MagicMock()
    petals_utils_misc_mock.DUMMY = torch.empty(0)

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.dht': hivemind_mock.dht,
        'hivemind.moe': hivemind_mock.moe,
        'hivemind.p2p': hivemind_mock.p2p,
        'hivemind.utils': hivemind_mock.utils,
        'tensor_parallel': tensor_parallel_mock,
        'petals.utils.misc': petals_utils_misc_mock
    }

    with mock.patch.dict('sys.modules', mocks):
        import transformers
        petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
        petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
        petals_mock.client = petals_client_mock

        with mock.patch.dict('sys.modules', {
            'petals': petals_mock,
            'petals.client': petals_client_mock
        }):
            import petals.client.ptune as ptune
            from transformers import PretrainedConfig

            class DummyModel(ptune.PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = mock.MagicMock()
                    self.word_embeddings.weight = torch.zeros(1, dtype=torch.float32)
                    self.init_prompts(config)

            config = PretrainedConfig()
            config.tuning_mode = "deep_ptune"
            config.pre_seq_len = 5
            config.hidden_size = 8
            config.num_hidden_layers = 4

            model = DummyModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert prompts.shape == (2, 5, 8)
            assert intermediate_prompts.shape == (3, 2, 5, 8)
