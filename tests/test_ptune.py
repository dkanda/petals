import sys
import os
sys.path.insert(0, os.path.abspath('src'))
import unittest.mock as mock

os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

with mock.patch.dict('sys.modules', {
    'petals.client.inference_session': mock.MagicMock(),
    'petals.client.remote_sequential': mock.MagicMock(),
    'petals.client.routing': mock.MagicMock(),
}):
    import hivemind
    hivemind.PeerID = mock.MagicMock()
    hivemind.MSGPackSerializer = mock.MagicMock()
    hivemind.get_logger = mock.MagicMock()

    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    import transformers.utils.import_utils
    transformers.utils.import_utils.is_torch_fx_available = lambda: False

    from petals.client.ptune import PTuneMixin
    from petals.utils.misc import DUMMY

    class MockModel(nn.Module, PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(10, config.hidden_size)
            self.init_prompts(config)

    def test_ptune():
        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=4,
            pre_seq_len=8,
            tuning_mode="ptune"
        )
        model = MockModel(config)
        assert model.pre_seq_len == 8

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert prompts.shape == (2, 8, 64)
        assert intermediate_prompts is DUMMY

    def test_deep_ptune():
        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=4,
            pre_seq_len=8,
            tuning_mode="deep_ptune"
        )
        model = MockModel(config)
        assert model.pre_seq_len == 8

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert prompts.shape == (2, 8, 64)
        assert intermediate_prompts.shape == (3, 2, 8, 64)

    test_ptune()
    test_deep_ptune()
