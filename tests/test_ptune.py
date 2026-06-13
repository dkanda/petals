import sys
import os
import torch
from unittest import mock

hivemind_mock = mock.MagicMock()
hivemind_mock.dht = mock.MagicMock()
hivemind_mock.moe = mock.MagicMock()
hivemind_mock.p2p = mock.MagicMock()
hivemind_mock.utils = mock.MagicMock()
hivemind_mock.utils.logging = mock.MagicMock()
hivemind_mock.proto = mock.MagicMock()
hivemind_mock.PeerID = hivemind_mock.p2p.PeerID
hivemind_mock.MSGPackSerializer = hivemind_mock.utils.MSGPackSerializer
hivemind_mock.get_logger = hivemind_mock.utils.get_logger

tensor_parallel_mock = mock.MagicMock()
petals_mock = mock.MagicMock()
petals_utils_misc_mock = mock.MagicMock()
petals_utils_misc_mock.DUMMY = torch.empty(0)

mocks = {
    'hivemind': hivemind_mock,
    'hivemind.dht': hivemind_mock.dht,
    'hivemind.moe': hivemind_mock.moe,
    'hivemind.p2p': hivemind_mock.p2p,
    'hivemind.utils': hivemind_mock.utils,
    'hivemind.utils.logging': hivemind_mock.utils.logging,
    'hivemind.proto': hivemind_mock.proto,
    'tensor_parallel': tensor_parallel_mock,
    'petals': petals_mock,
    'petals.utils': mock.MagicMock(),
    'petals.utils.misc': petals_utils_misc_mock,
}

def test_deep_ptune_shape():
    with mock.patch.dict('sys.modules', mocks):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["ptune"] = ptune
        sys.modules["petals.utils.misc"] = petals_utils_misc_mock
        spec.loader.exec_module(ptune)
        from transformers import PretrainedConfig

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32
                self.init_prompts(config)

        config = PretrainedConfig()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 5
        config.hidden_size = 16
        config.num_hidden_layers = 4

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert intermediate_prompts.shape == (3, 2, 5, 16), f"Expected (3, 2, 5, 16), got {intermediate_prompts.shape}"

if __name__ == "__main__":
    test_deep_ptune_shape()
