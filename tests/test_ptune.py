import pytest
import sys
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from unittest import mock
import hivemind
import hivemind.p2p
import hivemind.utils

# Polyfill hivemind things missing in top level package in this version
hivemind.PeerID = hivemind.p2p.PeerID
hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
hivemind.get_logger = hivemind.utils.get_logger

with mock.patch.dict('sys.modules', {
    'petals.client.inference_session': mock.MagicMock(),
    'petals.client.remote_sequential': mock.MagicMock(),
    'petals.client.routing': mock.MagicMock(),
}):
    from petals.client.ptune import PTuneMixin

def test_ptune_intermediate_prompts_shape():
    class DummyModel(nn.Module, PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=3
    )

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == torch.Size([2, 5, 16])
    assert intermediate_prompts.shape == torch.Size([2, 2, 5, 16])
