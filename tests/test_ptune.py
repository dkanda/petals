import sys
import os
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_intermediate_shape():
    with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
        import torch
        import torch.nn as nn
        from transformers import PretrainedConfig

        import hivemind.utils
        hivemind.get_logger = hivemind.utils.get_logger
        hivemind.PeerID = mock.MagicMock()

        sys.modules['petals.client.inference_session'] = mock.MagicMock()
        sys.modules['petals.client.remote_sequential'] = mock.MagicMock()
        sys.modules['petals.client.routing'] = mock.MagicMock()

        from petals.client.ptune import PTuneMixin, PTuneConfig

        class DummyConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.hidden_size = 16
                self.num_hidden_layers = 4
                self.pre_seq_len = 5
                self.tuning_mode = "deep_ptune"

        class DummyModel(PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        config = DummyConfig()
        model = DummyModel(config)
        prompts, intermediate = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 16])
        assert intermediate.shape == torch.Size([3, 2, 5, 16])
