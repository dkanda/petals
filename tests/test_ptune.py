import sys
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from transformers import PretrainedConfig

class MockPackage(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__path__ = []
        self.__spec__ = None

sys.modules['hivemind'] = MockPackage()
sys.modules['hivemind.moe'] = MockPackage()
sys.modules['hivemind.moe.client'] = MockPackage()
sys.modules['hivemind.moe.client.remote_expert_worker'] = MockPackage()
sys.modules['hivemind.moe.expert_uid'] = MockPackage()
sys.modules['hivemind.moe.server'] = MockPackage()
sys.modules['hivemind.moe.server.connection_handler'] = MockPackage()
sys.modules['hivemind.moe.server.module_backend'] = MockPackage()
sys.modules['hivemind.p2p'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.control'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon'] = MockPackage()
sys.modules['hivemind.proto'] = MockPackage()
sys.modules['hivemind.utils'] = MockPackage()
sys.modules['hivemind.utils.tensor_descr'] = MockPackage()
sys.modules['hivemind.utils.logging'] = MockPackage()
sys.modules['hivemind.utils.asyncio'] = MockPackage()
sys.modules['hivemind.utils.streaming'] = MockPackage()
sys.modules['hivemind.utils.mpfuture'] = MockPackage()
sys.modules['hivemind.utils.nested'] = MockPackage()
sys.modules['hivemind.compression'] = MockPackage()
sys.modules['hivemind.compression.serialization'] = MockPackage()
sys.modules['hivemind.compression.base'] = MockPackage()
sys.modules['hivemind.dht'] = MockPackage()
sys.modules['hivemind.dht.node'] = MockPackage()
sys.modules['tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.slicing_configs'] = MockPackage()

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_initialization():
    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )
    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert not hasattr(model, "intermediate_prompt_embeddings")

    # Check shape of prompt_embeddings
    assert model.prompt_embeddings.weight.shape == (5, 16)

def test_deep_ptune_initialization():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )
    model = MockModel(config)

    assert hasattr(model, "prompt_embeddings")
    assert hasattr(model, "intermediate_prompt_embeddings")

    # Check shape of intermediate_prompt_embeddings (pre_seq_len, (num_hidden_layers - 1) * hidden_size)
    assert model.intermediate_prompt_embeddings.weight.shape == (5, 3 * 16)

def test_get_prompt_ptune():
    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, 5, 16)
    assert intermediate_prompts is DUMMY

def test_get_prompt_deep_ptune():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check prompts shape
    assert prompts.shape == (batch_size, 5, 16)

    # Check intermediate_prompts shape
    assert intermediate_prompts.shape == (4, batch_size, 5, 16)

    # The first layer should be zero-padded
    assert torch.all(intermediate_prompts[0] == 0)

    # The remaining layers should not be zero-padded (with high probability, since weights are randomly initialized)
    assert not torch.all(intermediate_prompts[1:] == 0)
