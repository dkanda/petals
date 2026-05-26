import sys
import os
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

def test_deep_ptune():
    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    hivemind_mock = mock.MagicMock()
    hivemind_p2p = mock.MagicMock()
    hivemind_utils = mock.MagicMock()

    hivemind_mock.p2p = hivemind_p2p
    hivemind_mock.utils = hivemind_utils

    class PeerID:
        pass
    hivemind_mock.PeerID = PeerID
    hivemind_mock.p2p.PeerID = PeerID

    hivemind_mock.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.utils.MSGPackSerializer = hivemind_mock.MSGPackSerializer

    hivemind_mock.get_logger = mock.MagicMock()
    hivemind_mock.utils.get_logger = hivemind_mock.get_logger

    petals_mock = mock.MagicMock()
    petals_mock.__path__ = ["src/petals"]
    petals_mock.__spec__ = None

    petals_client_mock = mock.MagicMock()
    petals_client_mock.__path__ = ["src/petals/client"]
    petals_client_mock.__spec__ = None

    petals_mock.client = petals_client_mock

    petals_client_inference_session = mock.MagicMock()
    petals_client_inference_session.__path__ = []
    petals_client_inference_session.__spec__ = None

    petals_client_remote_sequential = mock.MagicMock()
    petals_client_remote_sequential.__path__ = []
    petals_client_remote_sequential.__spec__ = None

    petals_client_routing = mock.MagicMock()
    petals_client_routing.__path__ = []
    petals_client_routing.__spec__ = None

    petals_utils_misc = mock.MagicMock()
    petals_utils_misc.DUMMY = torch.empty(0)

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.p2p': hivemind_p2p,
        'hivemind.utils': hivemind_utils,
        'petals.client.inference_session': petals_client_inference_session,
        'petals.client.remote_sequential': petals_client_remote_sequential,
        'petals.client.routing': petals_client_routing,
        'petals.utils.misc': petals_utils_misc,
        'petals': petals_mock,
    }
    with mock.patch.dict('sys.modules', mocks):
        import importlib.util

        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune_module = importlib.util.module_from_spec(spec)
        sys.modules["ptune"] = ptune_module
        spec.loader.exec_module(ptune_module)
        PTuneMixin = ptune_module.PTuneMixin

        class DummyConfig(PretrainedConfig):
            def __init__(self, tuning_mode=None, pre_seq_len=0, hidden_size=8, num_hidden_layers=4, **kwargs):
                super().__init__(**kwargs)
                self.tuning_mode = tuning_mode
                self.pre_seq_len = pre_seq_len
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers

        class DummyModel(PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = DummyConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=8, num_hidden_layers=4)
        with mock.patch.object(ptune_module, '_original_register_parameter', nn.Module.register_parameter):
            model = DummyModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)
            assert prompts.shape == (2, 5, 8)
            assert intermediate_prompts.shape == (3, 2, 5, 8)
