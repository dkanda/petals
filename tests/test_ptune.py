import unittest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from unittest.mock import MagicMock, patch

# Local package patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock hivemind to avoid PeerID and other complex dependency issues during unit tests
class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

mock_hivemind = MockPackage()
mock_hivemind.PeerID = MagicMock()
sys.modules['hivemind'] = mock_hivemind
sys.modules['hivemind.moe'] = MockPackage()
sys.modules['hivemind.moe.expert_uid'] = MockPackage()
sys.modules['hivemind.moe.client'] = MockPackage()
sys.modules['hivemind.moe.client.remote_expert_worker'] = MockPackage()
sys.modules['hivemind.moe.server'] = MockPackage()
sys.modules['hivemind.moe.server.connection_handler'] = MockPackage()
sys.modules['hivemind.moe.server.module_backend'] = MockPackage()
sys.modules['hivemind.utils'] = MockPackage()
sys.modules['hivemind.utils.nested'] = MockPackage()
sys.modules['hivemind.utils.tensor_descr'] = MockPackage()
sys.modules['hivemind.utils.logging'] = MockPackage()
sys.modules['hivemind.utils.asyncio'] = MockPackage()
sys.modules['hivemind.utils.streaming'] = MockPackage()
sys.modules['hivemind.utils.mpfuture'] = MockPackage()
sys.modules['hivemind.compression'] = MockPackage()
sys.modules['hivemind.compression.serialization'] = MockPackage()
sys.modules['hivemind.optim'] = MockPackage()
sys.modules['hivemind.dht'] = MockPackage()
sys.modules['hivemind.dht.node'] = MockPackage()
sys.modules['hivemind.dht.routing'] = MockPackage()
sys.modules['hivemind.p2p'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.control'] = MockPackage()
sys.modules['hivemind.proto'] = MockPackage()
sys.modules['tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.slicing_configs'] = MockPackage()
sys.modules['tensor_parallel.tensor_parallel'] = MockPackage()

from petals.client.ptune import PTuneMixin, PTuneConfig
from petals.utils.misc import DUMMY

class DummyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 16
        self.num_hidden_layers = 4
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 5

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

class TestPTuneMixin(unittest.TestCase):
    def test_deep_ptune(self):
        config = DummyConfig()
        model = MockModel(config)

        # Check embedding dimension
        self.assertEqual(
            model.intermediate_prompt_embeddings.embedding_dim,
            (config.num_hidden_layers - 1) * config.hidden_size,
        )

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        # check shape of prompt
        self.assertEqual(prompts.shape, (batch_size, config.pre_seq_len, config.hidden_size))

        # Check shapes of intermediate_prompts
        self.assertEqual(
            intermediate_prompts.shape,
            (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)
        )

        # check that first layer is zero
        first_layer = intermediate_prompts[0]
        self.assertTrue(torch.all(first_layer == 0))

        # check that second layer is not zero
        second_layer = intermediate_prompts[1]
        self.assertFalse(torch.all(second_layer == 0))

if __name__ == '__main__':
    unittest.main()
