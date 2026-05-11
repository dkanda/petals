import sys
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from unittest import mock

import pytest

def test_deep_ptune():
    with open("src/petals/client/ptune.py", "r") as f:
        ptune_code = f.read()

    # We need a dummy hivemind module context just for exec
    import types
    dummy_hivemind = types.ModuleType('hivemind')
    dummy_hivemind.get_logger = mock.Mock()
    sys.modules['hivemind'] = dummy_hivemind

    dummy_misc = types.ModuleType('petals.utils.misc')
    dummy_misc.DUMMY = mock.Mock()
    sys.modules['petals.utils.misc'] = dummy_misc

    dummy_petals_utils = types.ModuleType('petals.utils')
    sys.modules['petals.utils'] = dummy_petals_utils

    dummy_petals = types.ModuleType('petals')
    sys.modules['petals'] = dummy_petals

    # Execute the module code in a clean dictionary
    module_dict = {}
    exec(ptune_code, module_dict)

    PTuneMixin = module_dict['PTuneMixin']

    class DummyConfig(PretrainedConfig):
        def __init__(self, tuning_mode="deep_ptune", pre_seq_len=5, num_hidden_layers=3, hidden_size=8, **kwargs):
            super().__init__(**kwargs)
            self.tuning_mode = tuning_mode
            self.pre_seq_len = pre_seq_len
            self.num_hidden_layers = num_hidden_layers
            self.hidden_size = hidden_size

    class DummyModel(nn.Module, PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

    config = DummyConfig()
    model = DummyModel(config)

    expected_embedding_dim = (config.num_hidden_layers - 1) * config.hidden_size
    assert model.intermediate_prompt_embeddings.embedding_dim == expected_embedding_dim

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
