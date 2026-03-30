import sys
from unittest.mock import MagicMock

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

    def get_logger(self, *args, **kwargs):
        return MagicMock()

# Mock deep dependencies of hivemind and tensor_parallel so that petals.client can be imported
sys.modules['hivemind'] = MockPackage()
sys.modules['hivemind.moe'] = MockPackage()
sys.modules['hivemind.moe.client'] = MockPackage()
sys.modules['hivemind.moe.client.remote_expert_worker'] = MockPackage()
sys.modules['hivemind.moe.expert_uid'] = MockPackage()
sys.modules['hivemind.moe.server'] = MockPackage()
sys.modules['hivemind.moe.server.connection_handler'] = MockPackage()
sys.modules['hivemind.moe.server.module_backend'] = MockPackage()
sys.modules['hivemind.p2p'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.control'] = MockPackage()
sys.modules['hivemind.proto'] = MockPackage()
sys.modules['hivemind.utils'] = MockPackage()
sys.modules['hivemind.utils.tensor_descr'] = MockPackage()
sys.modules['hivemind.utils.logging'] = MockPackage()
sys.modules['hivemind.utils.asyncio'] = MockPackage()
sys.modules['hivemind.utils.streaming'] = MockPackage()
sys.modules['hivemind.utils.mpfuture'] = MockPackage()
sys.modules['hivemind.utils.nested'] = MockPackage()
sys.modules['hivemind.compression'] = MockPackage()
sys.modules['hivemind.compression.quantization'] = MockPackage()
sys.modules['hivemind.compression.base'] = MockPackage()
sys.modules['hivemind.compression.serialization'] = MockPackage()
sys.modules['hivemind.dht'] = MockPackage()
sys.modules['hivemind.dht.node'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.datastructures'] = MockPackage()

sys.modules['tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.cross_device_ops'] = MockPackage()
sys.modules['tensor_parallel.slicer_wrapper'] = MockPackage()
sys.modules['tensor_parallel.tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.slicing_configs'] = MockPackage()

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from src.petals.client.ptune import PTuneMixin, PTuneConfig
from petals.utils.misc import DUMMY

class MockModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = MagicMock()
        self.word_embeddings.weight.device = torch.device('cpu')
        self.word_embeddings.weight.dtype = torch.float32
        self.init_prompts(config)

def test_deep_ptune():
    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 3
    config.hidden_size = 8
    config.num_hidden_layers = 4

    model = MockModel(config)

    assert model.intermediate_prompt_embeddings.weight.shape == (3, 3*8), f"Expected shape (3, 24), got {model.intermediate_prompt_embeddings.weight.shape}"
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert intermediate_prompts.shape == (4, 2, 3, 8), f"Expected shape (4, 2, 3, 8), got {intermediate_prompts.shape}"
    assert torch.all(intermediate_prompts[0] == 0), "Expected first layer prompt to be 0s"
