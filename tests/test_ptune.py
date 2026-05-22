import sys
import unittest.mock as mock
import torch
import os

os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'
sys.path.insert(0, os.path.abspath('src'))

from transformers import PretrainedConfig

def test_ptune_intermediate_prompt_shape():
    import transformers.utils.import_utils

    with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
        with mock.patch.dict('sys.modules', {
            'petals.client.inference_session': mock.MagicMock(__path__=[], __spec__=None),
            'petals.client.remote_sequential': mock.MagicMock(__path__=[], __spec__=None),
            'petals.client.routing': mock.MagicMock(__path__=[], __spec__=None)
        }):
            import hivemind
            hivemind.PeerID = hivemind.p2p.PeerID
            hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
            hivemind.get_logger = hivemind.utils.get_logger
            from petals.client.ptune import PTuneMixin

    class DummyModel(PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = mock.MagicMock()
            self.word_embeddings.weight.device = torch.device('cpu')
            self.word_embeddings.weight.dtype = torch.float32
            self.init_prompts(config)

    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=3
    )

    model = DummyModel(config)

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (2, 2, 5, 16) # (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)

if __name__ == '__main__':
    test_ptune_intermediate_prompt_shape()
    print("success!")