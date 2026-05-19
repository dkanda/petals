import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import unittest.mock as mock

def test_ptune_intermediate_prompts_shape():
    # Polyfill missing hivemind exports
    import hivemind
    import hivemind.p2p
    import hivemind.utils
    hivemind.PeerID = hivemind.p2p.PeerID
    hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
    hivemind.get_logger = hivemind.utils.get_logger

    # Mocking `transformers.utils.import_utils`
    import transformers
    import transformers.utils
    transformers.utils.import_utils.is_torch_fx_available = lambda: True

    with mock.patch.dict(sys.modules, {
        'petals.client.inference_session': mock.MagicMock(__path__=[], __spec__=None),
        'petals.client.remote_sequential': mock.MagicMock(__path__=[], __spec__=None),
        'petals.client.routing': mock.MagicMock(__path__=[], __spec__=None),
    }):
        from petals.client.ptune import PTuneMixin, PTuneConfig
        import petals.client.ptune as ptune

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)

                # Mock _original_register_parameter because mocking it at nn.Module breaks internal initialization
                with mock.patch.object(ptune, '_original_register_parameter', nn.Module.register_parameter):
                    self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=4
        )
        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 5, 16)
        # num_hidden_layers - 1 = 4 - 1 = 3
        assert intermediate_prompts.shape == (3, 2, 5, 16)
