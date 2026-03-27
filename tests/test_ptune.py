import sys
from unittest.mock import MagicMock

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

hivemind_mock = MockPackage()
sys.modules['hivemind'] = hivemind_mock
sys.modules['hivemind.moe'] = MockPackage()
sys.modules['hivemind.moe.client'] = MockPackage()
sys.modules['hivemind.moe.client.remote_expert_worker'] = MockPackage()
sys.modules['hivemind.moe.expert_uid'] = MockPackage()
sys.modules['hivemind.moe.server'] = MockPackage()
sys.modules['hivemind.moe.server.connection_handler'] = MockPackage()
sys.modules['hivemind.moe.server.module_backend'] = MockPackage()
sys.modules['hivemind.moe.server.task_pool'] = MockPackage()
sys.modules['hivemind.p2p'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.datastructures'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.control'] = MockPackage()
sys.modules['hivemind.dht'] = MockPackage()
sys.modules['hivemind.dht.node'] = MockPackage()
sys.modules['hivemind.proto'] = MockPackage()
sys.modules['hivemind.utils'] = MockPackage()
sys.modules['hivemind.utils.tensor_descr'] = MockPackage()
sys.modules['hivemind.utils.streaming'] = MockPackage()
sys.modules['hivemind.utils.asyncio'] = MockPackage()
sys.modules['hivemind.utils.logging'] = MockPackage()
sys.modules['hivemind.utils.logging.get_logger'] = MagicMock()
sys.modules['hivemind.utils.mpfuture'] = MockPackage()
sys.modules['hivemind.utils.nested'] = MockPackage()
sys.modules['hivemind.compression'] = MockPackage()
sys.modules['hivemind.compression.serialization'] = MockPackage()

tensor_parallel_mock = MockPackage()
sys.modules['tensor_parallel'] = tensor_parallel_mock
sys.modules['tensor_parallel.tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.slicing_configs'] = MockPackage()

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class DummyWordEmbeddings:
    def __init__(self, dtype=torch.float32, device='cpu'):
        self.weight = torch.zeros(1, dtype=dtype, device=device)

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = DummyWordEmbeddings(dtype=torch.float32)
        self.init_prompts(config)

def test_ptune_mode():
    config = PretrainedConfig()
    config.tuning_mode = "ptune"
    config.pre_seq_len = 5
    config.hidden_size = 16
    config.num_hidden_layers = 4

    model = DummyModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")
    assert model.prompt_embeddings.weight.shape == (5, 16)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, 5, 16)
    assert intermediate_prompts is DUMMY

def test_deep_ptune_mode():
    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5
    config.hidden_size = 16
    config.num_hidden_layers = 4

    model = DummyModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")

    # 4 layers -> intermediate needs (4-1) layers
    assert model.intermediate_prompt_embeddings.weight.shape == (5, 3 * 16)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, 5, 16)

    # The intermediate prompts must have shape (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (4, batch_size, 5, 16)

    # The first layer in intermediate_prompts must be zeros
    first_layer_prompts = intermediate_prompts[0]
    assert torch.all(first_layer_prompts == 0)
