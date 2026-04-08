import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from unittest import mock
import sys
import importlib.util

# We have to mock out petals to test ptune, since petals top level imports hivemind and transformers things that are missing or deep
class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def setup_module():
    sys.modules['hivemind'] = MockPackage()
    sys.modules['hivemind.utils'] = MockPackage()
    sys.modules['hivemind.utils.logging'] = MockPackage()

    mock_get_logger = mock.MagicMock()
    sys.modules['hivemind'].get_logger = mock_get_logger

def teardown_module():
    del sys.modules['hivemind']
    del sys.modules['hivemind.utils']
    del sys.modules['hivemind.utils.logging']

def get_ptune_module():
    with mock.patch.dict(sys.modules, {
        'hivemind': sys.modules['hivemind'],
        'hivemind.utils': sys.modules['hivemind.utils'],
        'hivemind.utils.logging': sys.modules['hivemind.utils.logging'],
    }):
        spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        # Provide petals.utils.misc.DUMMY directly
        dummy_mod = type(sys)('petals.utils.misc')
        dummy_mod.DUMMY = torch.empty(0)
        sys.modules['petals.utils.misc'] = dummy_mod

        spec.loader.exec_module(ptune)
        return ptune

def test_ptune_mixin_shapes():
    ptune = get_ptune_module()

    class MockModel(ptune.PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = nn.Embedding(10, config.hidden_size)
            self.init_prompts(config)

    # Test 'ptune'
    config_ptune = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )

    model1 = MockModel(config_ptune)
    prompts, intermediate_prompts = model1.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 16)
    assert "DUMMY" in repr(intermediate_prompts) or intermediate_prompts.numel() == 0

    # Test 'deep_ptune'
    config_deep_ptune = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )

    model2 = MockModel(config_deep_ptune)
    prompts2, intermediate_prompts2 = model2.get_prompt(batch_size=2)

    assert prompts2.shape == (2, 5, 16)
    # Shape should be (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts2.shape == (4, 2, 5, 16)
    # First layer should be zeros
    assert torch.all(intermediate_prompts2[0] == 0)
