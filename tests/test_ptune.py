import sys
import os
sys.path.insert(0, os.path.abspath('src'))
from unittest import mock

def test_deep_ptune_intermediate_shapes():
    import hivemind
    hivemind.PeerID = mock.MagicMock()
    hivemind.MSGPackSerializer = mock.MagicMock()
    hivemind.get_logger = mock.MagicMock()

    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_utils_misc_mock = mock.MagicMock(__path__=["src/petals/utils/misc"], __spec__=None)
    petals_utils_misc_mock.DUMMY = torch.empty(0)

    with mock.patch.dict('sys.modules', {
        'petals': petals_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
        'petals.client.inference_session': mock.MagicMock(),
        'petals.client.remote_sequential': mock.MagicMock(),
        'petals.client.routing': mock.MagicMock(),
    }):
        import petals.client.ptune as ptune
        PTuneMixin = ptune.PTuneMixin

    class MockConfig(PretrainedConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.tuning_mode = "deep_ptune"
            self.pre_seq_len = 5
            self.hidden_size = 16
            self.num_hidden_layers = 4

    class MockModel(PTuneMixin, nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

    config = MockConfig()
    model = MockModel(config)

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)
    assert prompts.shape == torch.Size([2, 5, 16])
    assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16])
