import pytest
import sys
import os
import torch
sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

from unittest import mock
import hivemind
hivemind.PeerID = hivemind.p2p.PeerID
hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
hivemind.get_logger = hivemind.utils.get_logger

from transformers import PretrainedConfig
import transformers.utils.import_utils
transformers.utils.import_utils.is_torch_fx_available = lambda: False

with mock.patch.dict('sys.modules', {
    'petals.client.inference_session': mock.MagicMock(),
    'petals.client.remote_sequential': mock.MagicMock(),
    'petals.client.routing': mock.MagicMock()
}):
    from petals.client.ptune import PTuneMixin

class DummyModel(PTuneMixin, torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4
    )
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (3, 2, 5, 16) # (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
