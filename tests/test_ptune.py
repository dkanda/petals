import sys
import os
import unittest
from unittest import mock
import torch
import torch.nn as nn

class TestPTune(unittest.TestCase):
    def test_ptune_intermediate_shapes(self):
        sys.path.insert(0, os.path.abspath('src'))

        petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
        petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
        petals_mock.client = petals_client_mock

        petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
        petals_utils_misc_mock = mock.MagicMock(__spec__=None)
        petals_utils_misc_mock.DUMMY = torch.empty(0)
        petals_utils_mock.misc = petals_utils_misc_mock
        petals_mock.utils = petals_utils_mock

        hivemind_mock = mock.MagicMock()
        hivemind_mock.utils = mock.MagicMock()
        hivemind_mock.p2p = mock.MagicMock()
        hivemind_mock.PeerID = hivemind_mock.p2p.PeerID
        hivemind_mock.MSGPackSerializer = hivemind_mock.utils.MSGPackSerializer
        hivemind_mock.get_logger = hivemind_mock.utils.get_logger

        with mock.patch.dict("sys.modules", {
            "petals": petals_mock,
            "petals.client": petals_client_mock,
            "petals.client.inference_session": mock.MagicMock(),
            "petals.client.remote_sequential": mock.MagicMock(),
            "petals.client.routing": mock.MagicMock(),
            "petals.utils": petals_utils_mock,
            "petals.utils.misc": petals_utils_misc_mock,
            "hivemind": hivemind_mock,
            "transformers": mock.MagicMock(),
        }):
            import petals.client.ptune as ptune

            class DummyConfig:
                tuning_mode = "deep_ptune"
                pre_seq_len = 5
                hidden_size = 10
                num_hidden_layers = 4

            class DummyModel(ptune.PTuneMixin):
                def __init__(self):
                    self.config = DummyConfig()
                    self.word_embeddings = nn.Embedding(100, 10)

            with mock.patch.object(ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
                model = DummyModel()
                model.init_prompts(model.config)

                # Check shapes
                self.assertEqual(model.intermediate_prompt_embeddings.weight.shape, torch.Size([5, 30]))

                prompts, intermediate_prompts = model.get_prompt(batch_size=2)
                self.assertEqual(intermediate_prompts.shape, torch.Size([3, 2, 5, 10]))

if __name__ == "__main__":
    unittest.main()
