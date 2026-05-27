import os
import sys

sys.path.insert(0, os.path.abspath('src'))

from unittest import mock
import torch
import torch.nn as nn

def test_ptune_intermediate_prompts_shape():
    hivemind_mock = mock.MagicMock()
    hivemind_mock.p2p.PeerID = mock.MagicMock()
    hivemind_mock.utils.MSGPackSerializer = mock.MagicMock()
    hivemind_mock.utils.get_logger = mock.MagicMock()
    hivemind_mock.PeerID = hivemind_mock.p2p.PeerID
    hivemind_mock.MSGPackSerializer = hivemind_mock.utils.MSGPackSerializer
    hivemind_mock.get_logger = hivemind_mock.utils.get_logger

    sys_modules_mocks = {
        'hivemind': hivemind_mock,
        'transformers': mock.MagicMock(),
        'petals.utils.misc': mock.MagicMock(DUMMY=torch.empty(0)),
    }

    with mock.patch.dict('sys.modules', sys_modules_mocks):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["ptune"] = ptune
        spec.loader.exec_module(ptune)

        class DummyConfig:
            tuning_mode = "deep_ptune"
            pre_seq_len = 5
            hidden_size = 16
            num_hidden_layers = 4

        class DummyModel(nn.Module, ptune.PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = DummyConfig()
        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 16])
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16])

if __name__ == "__main__":
    test_ptune_intermediate_prompts_shape()
