import torch
from unittest import mock
import sys
import os

# Insert the source directory so isolated imports work natively where possible
sys.path.insert(0, os.path.abspath('src'))

import importlib.util

def test_ptune_intermediate_prompt_embeddings_shape():
    """
    Tests that PTuneMixin's intermediate_prompt_embeddings in deep_ptune mode
    correctly sizes itself for (num_hidden_layers - 1) instead of num_hidden_layers.
    """
    with mock.patch.dict('sys.modules', {
        'hivemind': mock.MagicMock(),
        'hivemind.p2p': mock.MagicMock(),
        'hivemind.utils': mock.MagicMock(),
        'hivemind.utils.logging': mock.MagicMock(),
        'hivemind.dht': mock.MagicMock(),
        'hivemind.compression': mock.MagicMock(),
        'hivemind.moe': mock.MagicMock(),
        'hivemind.moe.client': mock.MagicMock(),
        'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
        'hivemind.proto': mock.MagicMock(),
        'tensor_parallel': mock.MagicMock(),
        'speedtest': mock.MagicMock(),
        'petals': mock.MagicMock(),
        'petals.utils': mock.MagicMock(),
        'petals.utils.misc': mock.MagicMock(),
    }):
        # Safely load ptune module directly bypassing init files that have unresolvable deps locally
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["ptune"] = ptune

        # Initialize the DUMMY constant the module expects
        ptune.DUMMY = torch.empty(0)
        sys.modules["petals.utils.misc"].DUMMY = torch.empty(0)

        spec.loader.exec_module(ptune)

        PTuneMixin = ptune.PTuneMixin

        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight = torch.empty(0, dtype=torch.float32)
                self.init_prompts(config)

        config = mock.MagicMock()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 10
        config.num_hidden_layers = 12
        config.hidden_size = 64

        model = DummyModel(config)

        # Verify the weight parameter shape
        expected_size = (config.num_hidden_layers - 1) * config.hidden_size
        assert model.intermediate_prompt_embeddings.weight.shape == torch.Size([config.pre_seq_len, expected_size]), \
            f"Expected {expected_size} size, got {model.intermediate_prompt_embeddings.weight.shape}"

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        # Verify the view / returned tensor shape
        # Expected shape: [num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size]
        assert intermediate_prompts.shape == torch.Size([config.num_hidden_layers - 1, 2, config.pre_seq_len, config.hidden_size]), \
            f"Expected intermediate_prompts shape of [11, 2, 10, 64], got {intermediate_prompts.shape}"
