import sys
import os
from unittest import mock

import torch
from transformers import PretrainedConfig

hivemind_mock = mock.MagicMock()
hivemind_mock.p2p = mock.MagicMock()
hivemind_mock.utils = mock.MagicMock()
hivemind_mock.utils.logging = mock.MagicMock()
hivemind_mock.PeerID = hivemind_mock.p2p.PeerID
hivemind_mock.MSGPackSerializer = hivemind_mock.utils.MSGPackSerializer
hivemind_mock.get_logger = hivemind_mock.utils.get_logger
hivemind_mock.dht = mock.MagicMock()
hivemind_mock.moe = mock.MagicMock()
hivemind_mock.proto = mock.MagicMock()

tensor_parallel_mock = mock.MagicMock()
petals_mock = mock.MagicMock()
petals_client_mock = mock.MagicMock()
petals_client_mock.__path__ = ["src/petals/client"]
petals_client_mock.__spec__ = None
petals_mock.client = petals_client_mock

# Also dummy module for petals.utils.misc
petals_utils_misc_mock = mock.MagicMock()
petals_utils_misc_mock.DUMMY = torch.empty(0)

mocks = {
    'hivemind': hivemind_mock,
    'tensor_parallel': tensor_parallel_mock,
    'petals.client.inference_session': mock.MagicMock(),
    'petals.client.remote_sequential': mock.MagicMock(),
    'petals.client.routing': mock.MagicMock(),
    'petals': petals_mock,
    'petals.client': petals_client_mock,
    'petals.utils': mock.MagicMock(),
    'petals.utils.misc': petals_utils_misc_mock,
}

with mock.patch.dict('sys.modules', mocks):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)
    sys.modules["ptune"] = ptune
    spec.loader.exec_module(ptune)

def test_ptune_mixin_deep_ptune():
    class DummyModel(ptune.PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.init_prompts(config)
            self.word_embeddings = mock.MagicMock()
            self.word_embeddings.weight.device = torch.device('cpu')
            self.word_embeddings.weight.dtype = torch.float32

    config = PretrainedConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=4)
    model = DummyModel(config)

    assert model.intermediate_prompt_embeddings.weight.shape == torch.Size([5, 48])

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16])
