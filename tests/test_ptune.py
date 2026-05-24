import sys
import os

sys.path.insert(0, os.path.abspath('src'))

os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

import torch
import torch.nn as nn
from unittest import mock
import pytest

# Mock nested packages
class MockModule(mock.MagicMock):
    __path__ = []
    __spec__ = None

def test_ptune_intermediate_prompts_shape():
    petals_mock = MockModule()
    petals_client_mock = MockModule()
    petals_mock.client = petals_client_mock

    # Put heavy dependencies out of the way for standard isolated testing
    sys_modules_patches = {
        'petals.client.inference_session': MockModule(),
        'petals.client.remote_sequential': MockModule(),
        'petals.client.routing': MockModule(),
    }

    with mock.patch.dict('sys.modules', sys_modules_patches):
        # Polyfill top-level exports for ptune/hivemind
        import hivemind
        hivemind.PeerID = hivemind.p2p.PeerID
        hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
        hivemind.get_logger = hivemind.utils.get_logger

        with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
            import petals.client.ptune as ptune
            from transformers import PretrainedConfig

            class DummyModel(ptune.PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            config = PretrainedConfig(
                hidden_size=64,
                num_hidden_layers=10,
                tuning_mode="deep_ptune",
                pre_seq_len=5
            )

            # Test after change
            model = DummyModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert prompts.shape == torch.Size([2, 5, 64])
            assert intermediate_prompts.shape == torch.Size([9, 2, 5, 64])
