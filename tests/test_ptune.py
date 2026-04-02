import unittest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

import sys
from unittest import mock
import importlib.util

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

class TestPTune(unittest.TestCase):
    def setUp(self):
        mock_modules = {
            'hivemind': MockPackage(),
            'hivemind.moe': MockPackage(),
            'hivemind.moe.client': MockPackage(),
            'hivemind.moe.client.remote_expert_worker': MockPackage(),
            'hivemind.moe.server': MockPackage(),
            'hivemind.moe.server.module_backend': MockPackage(),
            'hivemind.moe.expert_uid': MockPackage(),
            'hivemind.utils': MockPackage(),
            'hivemind.utils.nested': MockPackage(),
            'hivemind.dht': MockPackage(),
            'hivemind.dht.routing': MockPackage(),
            'hivemind.p2p': MockPackage(),
            'hivemind.p2p.p2p_daemon_bindings': MockPackage(),
            'hivemind.p2p.p2p_daemon_bindings.datastructures': MockPackage(),
            'hivemind.p2p.p2p_daemon': MockPackage(),
            'tensor_parallel': MockPackage(),
            'tensor_parallel.slicing_configs': MockPackage(),
            'petals': MockPackage(),
            'petals.utils.misc': type('obj', (object,), {'DUMMY': torch.empty(0)})()
        }

        self.patcher = mock.patch.dict('sys.modules', mock_modules)
        self.patcher.start()

        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        self.ptune_module = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = self.ptune_module
        spec.loader.exec_module(self.ptune_module)

        self.PTuneMixin = self.ptune_module.PTuneMixin

        class DummyModel(nn.Module, self.PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = type('obj', (object,), {'weight': torch.zeros(1, dtype=torch.float32)})()
                self.init_prompts(config)
        self.DummyModel = DummyModel

    def tearDown(self):
        sys.modules.pop("petals.client.ptune", None)
        self.patcher.stop()

    def test_deep_ptune(self):
        config = PretrainedConfig()
        config.hidden_size = 16
        config.num_hidden_layers = 4
        config.pre_seq_len = 5
        config.tuning_mode = "deep_ptune"

        model = self.DummyModel(config)

        self.assertEqual(model.intermediate_prompt_embeddings.weight.shape, (5, 3 * 16))

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        self.assertEqual(prompts.shape, (2, 5, 16))
        self.assertEqual(intermediate_prompts.shape, (4, 2, 5, 16))

        self.assertTrue(torch.allclose(intermediate_prompts[0], torch.zeros(2, 5, 16)))

        self.assertFalse(torch.allclose(intermediate_prompts[1:], torch.zeros(3, 2, 5, 16)))

if __name__ == "__main__":
    unittest.main()
