import os
import sys
sys.path.insert(0, os.path.abspath('src'))

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from unittest import mock

with mock.patch.dict('sys.modules', {
    'petals.utils.misc': mock.MagicMock(DUMMY=torch.empty(0)),
    'hivemind': mock.MagicMock(),
}):
    # Import PTuneMixin directly, bypassing petals.__init__ completely
    import importlib.util
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)
    sys.modules["ptune"] = ptune
    spec.loader.exec_module(ptune)

    class DummyModel(nn.Module, ptune.PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(10, config.hidden_size)
            self.init_prompts(config)

def test_ptune_intermediate_shape():
    config = PretrainedConfig(tuning_mode='deep_ptune', pre_seq_len=5, hidden_size=16, num_hidden_layers=4)
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (3, 2, 5, 16)
