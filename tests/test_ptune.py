import sys
import os
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

class TestPTune(unittest.TestCase):
    def test_ptune_intermediate_shape(self):
        hivemind_mock = mock.MagicMock()
        hivemind_mock.moe = mock.MagicMock()
        hivemind_mock.p2p = mock.MagicMock()
        hivemind_mock.utils = mock.MagicMock()
        hivemind_mock.utils.logging = mock.MagicMock()
        hivemind_mock.proto = mock.MagicMock()

        # Polyfill
        hivemind_mock.PeerID = hivemind_mock.p2p.PeerID
        hivemind_mock.MSGPackSerializer = hivemind_mock.utils.MSGPackSerializer
        hivemind_mock.get_logger = hivemind_mock.utils.get_logger

        mocks = {
            'hivemind': hivemind_mock,
            'hivemind.dht': mock.MagicMock(),
            'hivemind.moe': hivemind_mock.moe,
            'hivemind.moe.expert_uid': mock.MagicMock(),
            'hivemind.moe.client': mock.MagicMock(),
            'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
            'hivemind.p2p': hivemind_mock.p2p,
            'hivemind.p2p.p2p_daemon_bindings': mock.MagicMock(),
            'hivemind.p2p.p2p_daemon_bindings.datastructures': mock.MagicMock(),
            'hivemind.utils': hivemind_mock.utils,
            'hivemind.utils.logging': hivemind_mock.utils.logging,
            'hivemind.proto': hivemind_mock.proto,
            'tensor_parallel': mock.MagicMock(),
            'petals.client.inference_session': mock.MagicMock(),
            'petals.client.remote_sequential': mock.MagicMock(),
            'petals.client.routing': mock.MagicMock(),
        }

        with mock.patch.dict('sys.modules', mocks):
            import petals.client.ptune as ptune
            import torch
            from transformers import PretrainedConfig

            class DummyModel(ptune.PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.tuning_mode = config.tuning_mode
                    self.word_embeddings = mock.MagicMock()
                    self.word_embeddings.weight.device = torch.device('cpu')
                    self.word_embeddings.weight.dtype = torch.float32
                    self.init_prompts(config)

            config = PretrainedConfig(
                tuning_mode="deep_ptune",
                pre_seq_len=5,
                hidden_size=10,
                num_hidden_layers=3
            )

            model = DummyModel(config)

            self.assertEqual(model.intermediate_prompt_embeddings.weight.shape, torch.Size([5, 20]))

            prompts, intermediate_prompts = model.get_prompt(batch_size=2)
            self.assertEqual(intermediate_prompts.shape, torch.Size([2, 2, 5, 10]))

if __name__ == '__main__':
    unittest.main()
