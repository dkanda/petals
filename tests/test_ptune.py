import os
import sys
import torch
import torch.nn as nn
from unittest import mock

def test_ptune_mixin_intermediate_prompts_shape():
    import transformers.utils.import_utils
    transformers.utils.import_utils.is_torch_fx_available = lambda: False

    import hivemind
    import hivemind.p2p
    import hivemind.utils
    hivemind.PeerID = hivemind.p2p.PeerID
    hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer
    hivemind.get_logger = hivemind.utils.get_logger

    mock_is = mock.MagicMock()
    mock_is.__path__ = []
    mock_is.__spec__ = None

    mock_rs = mock.MagicMock()
    mock_rs.__path__ = []
    mock_rs.__spec__ = None

    mock_rt = mock.MagicMock()
    mock_rt.__path__ = []
    mock_rt.__spec__ = None

    with mock.patch.dict("sys.modules", {
        "petals.client.inference_session": mock_is,
        "petals.client.remote_sequential": mock_rs,
        "petals.client.routing": mock_rt,
    }):
        from petals.client.ptune import PTuneMixin
        import petals.client.ptune as ptune_module

        class DummyConfig:
            def __init__(self, pre_seq_len, hidden_size, num_hidden_layers, tuning_mode):
                self.pre_seq_len = pre_seq_len
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers
                self.tuning_mode = tuning_mode

        config = DummyConfig(pre_seq_len=10, hidden_size=32, num_hidden_layers=5, tuning_mode="deep_ptune")

        with mock.patch.object(ptune_module, '_original_register_parameter', torch.nn.Module.register_parameter):
            mixin = PTuneMixin()
            mixin.word_embeddings = nn.Embedding(100, 32)
            mixin.config = config
            mixin.init_prompts(config)

            prompts, intermediate_prompts = mixin.get_prompt(batch_size=2)
            assert prompts.shape == torch.Size([2, 10, 32])
            assert intermediate_prompts.shape == torch.Size([4, 2, 10, 32])
