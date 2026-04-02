import unittest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import sys
from unittest.mock import Mock, patch

# Mock modules to avoid heavy dependencies during pytest collection
class MockPackage(Mock):
    __path__ = []
    __spec__ = None

mock_modules = {
    'hivemind': MockPackage(),
    'hivemind.p2p': MockPackage(),
    'hivemind.p2p.p2p_daemon': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings.control': MockPackage(),
    'hivemind.moe': MockPackage(),
    'hivemind.moe.expert_uid': MockPackage(),
    'hivemind.moe.server': MockPackage(),
    'hivemind.moe.server.connection_handler': MockPackage(),
    'hivemind.moe.server.module_backend': MockPackage(),
    'hivemind.moe.client': MockPackage(),
    'hivemind.moe.client.remote_expert_worker': MockPackage(),
    'hivemind.utils': MockPackage(),
    'hivemind.utils.asyncio': MockPackage(),
    'hivemind.utils.nested': MockPackage(),
    'hivemind.utils.tensor_descr': MockPackage(),
    'hivemind.utils.logging': MockPackage(),
    'hivemind.utils.streaming': MockPackage(),
    'hivemind.utils.mpfuture': MockPackage(),
    'hivemind.dht': MockPackage(),
    'hivemind.dht.node': MockPackage(),
    'hivemind.compression': MockPackage(),
    'hivemind.compression.serialization': MockPackage(),
    'hivemind.dht.routing': MockPackage(),
    'hivemind.proto': MockPackage(),
    'hivemind.proto.runtime_pb2': MockPackage(),
    'tensor_parallel': MockPackage(),
    'tensor_parallel.slicing_configs': MockPackage(),
    'tensor_parallel.tensor_parallel': MockPackage(),
    'speedtest': MockPackage(),
}

# Make sure patcher runs before any import
patcher = patch.dict('sys.modules', mock_modules)
patcher.start()

def setup_module():
    global patcher_dict
    patcher_dict = patcher

def teardown_module():
    patcher_dict.stop()

from petals.client.ptune import PTuneMixin, PTuneConfig
from petals.utils.misc import DUMMY

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

class TestPTune(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 64
        self.num_hidden_layers = 4
        self.pre_seq_len = 5
        self.batch_size = 2

        self.config_ptune = PretrainedConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            tuning_mode="ptune",
            pre_seq_len=self.pre_seq_len
        )

        self.config_deep_ptune = PretrainedConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            tuning_mode="deep_ptune",
            pre_seq_len=self.pre_seq_len
        )

    def test_ptune_initialization(self):
        model = DummyModel(self.config_ptune)
        self.assertTrue(hasattr(model, 'prompt_embeddings'))
        self.assertFalse(hasattr(model, 'intermediate_prompt_embeddings'))

        self.assertEqual(model.prompt_embeddings.weight.shape, (self.pre_seq_len, self.hidden_size))

    def test_deep_ptune_initialization(self):
        model = DummyModel(self.config_deep_ptune)
        self.assertTrue(hasattr(model, 'prompt_embeddings'))
        self.assertTrue(hasattr(model, 'intermediate_prompt_embeddings'))

        self.assertEqual(model.prompt_embeddings.weight.shape, (self.pre_seq_len, self.hidden_size))

        # Check that intermediate_prompt_embeddings holds params for num_hidden_layers - 1
        expected_size = (self.num_hidden_layers - 1) * self.hidden_size
        self.assertEqual(model.intermediate_prompt_embeddings.weight.shape, (self.pre_seq_len, expected_size))

    def test_get_prompt_ptune(self):
        model = DummyModel(self.config_ptune)
        prompts, intermediate_prompts = model.get_prompt(self.batch_size)

        self.assertEqual(prompts.shape, (self.batch_size, self.pre_seq_len, self.hidden_size))

        # In ptune, intermediate_prompts should be DUMMY
        self.assertTrue(intermediate_prompts is DUMMY or torch.allclose(intermediate_prompts, DUMMY))

    def test_get_prompt_deep_ptune(self):
        model = DummyModel(self.config_deep_ptune)
        prompts, intermediate_prompts = model.get_prompt(self.batch_size)

        self.assertEqual(prompts.shape, (self.batch_size, self.pre_seq_len, self.hidden_size))

        # In deep_ptune, intermediate_prompts should be formatted correctly
        # permute order is [2, 0, 1, 3] -> [num_hidden_layers, batch_size, pre_seq_len, hidden_size]
        self.assertEqual(
            intermediate_prompts.shape,
            (self.num_hidden_layers, self.batch_size, self.pre_seq_len, self.hidden_size)
        )

        # Check that the first layer is zero-padded
        first_layer_prompts = intermediate_prompts[0, :, :, :]
        self.assertTrue(torch.allclose(first_layer_prompts, torch.zeros_like(first_layer_prompts)))

        # Check that the remaining layers are not (necessarily) zero
        # We can just verify the shape of the remaining part
        remaining_layers = intermediate_prompts[1:, :, :, :]
        self.assertEqual(
            remaining_layers.shape,
            (self.num_hidden_layers - 1, self.batch_size, self.pre_seq_len, self.hidden_size)
        )
