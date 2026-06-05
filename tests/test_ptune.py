import os
import sys

sys.path.insert(0, os.path.abspath('src'))
from unittest import mock
import torch

os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

def test_ptune_intermediate_prompts_shape():
    import hivemind
    hivemind.PeerID = mock.MagicMock()
    hivemind.MSGPackSerializer = mock.MagicMock()
    hivemind.get_logger = mock.MagicMock()

    with mock.patch("transformers.utils.import_utils.is_torch_fx_available", return_value=False, create=True), \
         mock.patch.dict("sys.modules", {
            "petals.client.inference_session": mock.MagicMock(),
            "petals.client.remote_sequential": mock.MagicMock(),
            "petals.client.routing": mock.MagicMock(),
         }):
        from petals.client.ptune import PTuneMixin
        from transformers import PretrainedConfig

        class MockConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.pre_seq_len = 5
                self.tuning_mode = "deep_ptune"
                self.hidden_size = 16
                self.num_hidden_layers = 4

        class DummyModel(torch.nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        model = DummyModel(MockConfig())
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 16])
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16])
