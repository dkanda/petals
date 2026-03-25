import sys
import unittest
from unittest.mock import MagicMock

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

hivemind_mock = MockPackage()

# Comprehensive hivemind mock
sys.modules['hivemind'] = hivemind_mock
for submod in [
    'moe', 'moe.client', 'moe.client.remote_expert_worker', 'moe.expert_uid',
    'moe.server', 'moe.server.connection_handler', 'moe.server.module_backend',
    'p2p', 'p2p.p2p_daemon', 'p2p.p2p_daemon_bindings', 'p2p.p2p_daemon_bindings.datastructures', 'p2p.p2p_daemon_bindings.control',
    'proto',
    'utils', 'utils.asyncio', 'utils.logging', 'utils.nested', 'utils.tensor_deserializer',
    'utils.tensor_descr', 'utils.streaming', 'utils.mpfuture',
    'compression', 'compression.base', 'compression.serialization',
    'dht', 'dht.node',
    'optim', 'optim.performance_ema'
]:
    sys.modules[f'hivemind.{submod}'] = hivemind_mock

sys.modules['dijkstar'] = MockPackage()
sys.modules['tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.slicing_configs'] = MockPackage()
sys.modules['tensor_parallel.tensor_parallel'] = MockPackage()
sys.modules['speedtest'] = MockPackage()
sys.modules['pydantic'] = MockPackage()
sys.modules['pydantic.v1'] = MockPackage()
sys.modules['fastapi'] = MockPackage()
sys.modules['uvicorn'] = MockPackage()
sys.modules['sse_starlette'] = MockPackage()
sys.modules['cpufeature'] = MockPackage()

import torch
from transformers import PretrainedConfig

class MockConfig(PretrainedConfig):
    def __init__(self, pre_seq_len=0, tuning_mode=None, num_hidden_layers=3, hidden_size=4, **kwargs):
        super().__init__(**kwargs)
        self.pre_seq_len = pre_seq_len
        self.tuning_mode = tuning_mode
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class DummyModel(PTuneMixin, torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = MagicMock()
        self.word_embeddings.weight = MagicMock()
        self.word_embeddings.weight.device = torch.device('cpu')
        self.word_embeddings.weight.dtype = torch.float32

class TestPTune(unittest.TestCase):
    def test_ptune_init_prompts(self):
        config = MockConfig(pre_seq_len=5, tuning_mode="ptune", num_hidden_layers=3, hidden_size=8)
        model = DummyModel(config)
        model.init_prompts(config)

        self.assertEqual(model.pre_seq_len, 5)
        self.assertTrue(hasattr(model, "prompt_embeddings"))
        self.assertEqual(model.prompt_embeddings.weight.shape, (5, 8))
        self.assertFalse(hasattr(model, "intermediate_prompt_embeddings"))

    def test_deep_ptune_init_prompts(self):
        config = MockConfig(pre_seq_len=5, tuning_mode="deep_ptune", num_hidden_layers=4, hidden_size=8)
        model = DummyModel(config)
        model.init_prompts(config)

        self.assertEqual(model.pre_seq_len, 5)
        self.assertTrue(hasattr(model, "prompt_embeddings"))
        self.assertEqual(model.prompt_embeddings.weight.shape, (5, 8))
        self.assertTrue(hasattr(model, "intermediate_prompt_embeddings"))
        self.assertEqual(model.intermediate_prompt_embeddings.weight.shape, (5, (4 - 1) * 8))

    def test_get_prompt_deep_ptune(self):
        batch_size = 2
        pre_seq_len = 5
        num_hidden_layers = 4
        hidden_size = 8

        config = MockConfig(pre_seq_len=pre_seq_len, tuning_mode="deep_ptune", num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        model = DummyModel(config)
        model.init_prompts(config)

        prompts, intermediate_prompts = model.get_prompt(batch_size)

        self.assertEqual(prompts.shape, (batch_size, pre_seq_len, hidden_size))
        self.assertEqual(intermediate_prompts.shape, (num_hidden_layers, batch_size, pre_seq_len, hidden_size))
        self.assertTrue(torch.all(intermediate_prompts[0] == 0))

    def test_get_prompt_ptune(self):
        batch_size = 2
        pre_seq_len = 5
        num_hidden_layers = 4
        hidden_size = 8

        config = MockConfig(pre_seq_len=pre_seq_len, tuning_mode="ptune", num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        model = DummyModel(config)
        model.init_prompts(config)

        prompts, intermediate_prompts = model.get_prompt(batch_size)

        self.assertEqual(prompts.shape, (batch_size, pre_seq_len, hidden_size))
        self.assertIs(intermediate_prompts, DUMMY)

if __name__ == '__main__':
    unittest.main()
