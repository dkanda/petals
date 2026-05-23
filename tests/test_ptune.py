import os
import sys
import unittest.mock as mock

# setup path
sys.path.insert(0, os.path.abspath("src"))
import torch
import torch.nn as nn
from transformers import PretrainedConfig

os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

with mock.patch.dict("sys.modules", {
    "petals.client.inference_session": mock.MagicMock(__path__=[], __spec__=None),
    "petals.client.remote_sequential": mock.MagicMock(__path__=[], __spec__=None),
    "petals.client.routing": mock.MagicMock(__path__=[], __spec__=None)
}), mock.patch("transformers.utils.import_utils.is_torch_fx_available", return_value=False, create=True):

    import hivemind.p2p
    import hivemind.utils
    hivemind.PeerID = hivemind.p2p.PeerID
    hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
    hivemind.get_logger = hivemind.utils.get_logger

    from petals.client.ptune import PTuneMixin, PTuneConfig

class DummyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 8
        self.hidden_size = 16
        self.num_hidden_layers = 4

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_ptune_shapes():
    config = DummyConfig()
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 8, 16)
    assert intermediate_prompts.shape == (3, 2, 8, 16)

if __name__ == "__main__":
    test_ptune_shapes()
