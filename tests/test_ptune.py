import os
import sys
sys.path.insert(0, os.path.abspath('src'))
from unittest import mock

class DummyConfig:
    def __init__(self):
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 5
        self.hidden_size = 16
        self.num_hidden_layers = 4

def test_ptune_intermediate_prompts_shape():
    import torch
    import torch.nn as nn
    import hivemind

    hivemind.PeerID = hivemind.p2p.PeerID
    hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
    hivemind.get_logger = hivemind.utils.get_logger
    os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

    with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
        with mock.patch.dict('sys.modules', {
            'petals.client.inference_session': mock.MagicMock(),
            'petals.client.remote_sequential': mock.MagicMock(),
            'petals.client.routing': mock.MagicMock()
        }):
            import petals.client.ptune as ptune

            class DummyModel(nn.Module, ptune.PTuneMixin):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = nn.Embedding(100, config.hidden_size)
                    self.init_prompts(config)

            config = DummyConfig()
            with mock.patch.object(ptune, '_original_register_parameter', nn.Module.register_parameter):
                model = DummyModel(config)
                prompts, intermediate_prompts = model.get_prompt(batch_size=2)

                assert prompts.shape == (2, 5, 16)
                # Should be num_hidden_layers - 1 (which is 4 - 1 = 3)
                assert intermediate_prompts.shape == (3, 2, 5, 16)
