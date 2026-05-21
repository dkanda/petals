import sys
import os

sys.path.insert(0, os.path.abspath('src'))

import unittest.mock as mock
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers import PretrainedConfig

import hivemind
from hivemind.p2p import PeerID
from hivemind.utils import MSGPackSerializer, get_logger

# Polyfill missing hivemind exports
hivemind.PeerID = PeerID
hivemind.MSGPackSerializer = MSGPackSerializer
hivemind.get_logger = get_logger

import os
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

import transformers.utils.import_utils
transformers.utils.import_utils.is_torch_fx_available = lambda: False

def test_ptune_intermediate_prompt_shape():
    with mock.patch.dict(
        "sys.modules",
        {
            "petals.client.inference_session": mock.MagicMock(),
            "petals.client.remote_sequential": mock.MagicMock(),
            "petals.client.routing": mock.MagicMock(),
        },
    ):
        from petals.client.ptune import PTuneMixin, PTuneConfig
        import petals.client.ptune as ptune_module

        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(100, 10)
                self.init_prompts(config)

        config = PretrainedConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=10, num_hidden_layers=3)

        with mock.patch.object(ptune_module, '_original_register_parameter', nn.Module.register_parameter):
            model = DummyModel(config)

        assert model.intermediate_prompt_embeddings.weight.shape == (5, 20)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert prompts.shape == (2, 5, 10)
        assert intermediate_prompts.shape == (2, 2, 5, 10)

if __name__ == "__main__":
    test_ptune_intermediate_prompt_shape()
