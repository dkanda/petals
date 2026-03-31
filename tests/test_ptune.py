import sys
from unittest.mock import MagicMock
import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

# Mock packages before importing PTuneMixin
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
sys.modules['hivemind.p2p'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.control'] = MockPackage()
sys.modules['hivemind.proto'] = MockPackage()
sys.modules['hivemind.utils'] = MockPackage()
sys.modules['hivemind.utils.nested'] = MockPackage()
sys.modules['hivemind.utils.tensor_descr'] = MockPackage()
sys.modules['hivemind.utils.mpfuture'] = MockPackage()
sys.modules['hivemind.compression'] = MockPackage()
sys.modules['hivemind.compression.quantization'] = MockPackage()
sys.modules['hivemind.moe.server.module_backend'] = MockPackage()
sys.modules['hivemind.moe.server.task_pool'] = MockPackage()
sys.modules['hivemind.utils.streaming'] = MockPackage()
sys.modules['hivemind.dht'] = MockPackage()
sys.modules['hivemind.dht.routing'] = MockPackage()
sys.modules['hivemind.dht.node'] = MockPackage()
sys.modules['hivemind.dht.crypto'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon'] = MockPackage()
sys.modules['hivemind.utils.logging'] = MockPackage()
sys.modules['hivemind.utils.networking'] = MockPackage()
sys.modules['hivemind.utils.performance_ema'] = MockPackage()
sys.modules['hivemind.utils.asyncio'] = MockPackage()
sys.modules['hivemind.utils.tensor_descr'] = MockPackage()
sys.modules['hivemind.moe.server.module_backend'] = MockPackage()
sys.modules['hivemind.moe.server.task_pool'] = MockPackage()
sys.modules['hivemind.utils.limits'] = MockPackage()
sys.modules['hivemind.utils.serializer'] = MockPackage()
sys.modules['hivemind.utils.timed_storage'] = MockPackage()
sys.modules['hivemind.compression.base'] = MockPackage()
sys.modules['hivemind.compression.serialization'] = MockPackage()
sys.modules['tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.slicing_configs'] = MockPackage()
sys.modules['tensor_parallel.tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.cross_device_ops'] = MockPackage()

# Now import the module
from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_mixin_init_prompts_deep_ptune():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=3
    )
    model = DummyModel(config)

    # Check that prompt_embeddings exist and are configured correctly
    assert hasattr(model, 'prompt_embeddings')
    assert model.prompt_embeddings.weight.shape == (5, 16)

    # Check that intermediate_prompt_embeddings exist and are configured for num_hidden_layers - 1
    assert hasattr(model, 'intermediate_prompt_embeddings')
    assert model.intermediate_prompt_embeddings.weight.shape == (5, 2 * 16)  # (3-1) * 16

def test_ptune_mixin_get_prompt_deep_ptune():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=3
    )
    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check prompts shape
    assert prompts.shape == (2, 5, 16)

    # Check intermediate_prompts shape
    assert intermediate_prompts.shape == (3, 2, 5, 16)  # (num_hidden_layers, batch_size, pre_seq_len, hidden_size)

    # Check that the first layer in intermediate_prompts is correctly zero-padded
    assert torch.all(intermediate_prompts[0] == 0)

def test_ptune_mixin_get_prompt_ptune():
    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=3
    )
    model = DummyModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check prompts shape
    assert prompts.shape == (2, 5, 16)

    # Check intermediate_prompts is DUMMY
    assert intermediate_prompts is DUMMY or torch.allclose(intermediate_prompts, DUMMY)
