import sys
import os
from unittest import mock
import torch

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_mixin():
    hivemind_mock = mock.MagicMock()
    hivemind_mock.get_logger = mock.MagicMock()
    hivemind_mock.p2p = mock.MagicMock()
    hivemind_mock.dht = mock.MagicMock()
    hivemind_mock.moe = mock.MagicMock()
    hivemind_mock.utils = mock.MagicMock()

    petals_mock = mock.MagicMock()
    petals_mock.client = mock.MagicMock()
    petals_mock.models = mock.MagicMock()
    petals_utils_mock = mock.MagicMock()
    petals_utils_mock.misc = mock.MagicMock()
    petals_utils_mock.misc.DUMMY = torch.empty(0)

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.p2p': hivemind_mock.p2p,
        'hivemind.dht': hivemind_mock.dht,
        'hivemind.moe': hivemind_mock.moe,
        'hivemind.utils': hivemind_mock.utils,
        'petals': petals_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_mock.misc,
    }

    with mock.patch.dict('sys.modules', mocks):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["ptune"] = ptune
        spec.loader.exec_module(ptune)

        class MockConfig:
            tuning_mode = "deep_ptune"
            pre_seq_len = 5
            num_hidden_layers = 10
            hidden_size = 16

        class MockModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight = torch.zeros(1, dtype=torch.float32)
                self.init_prompts(config)

        config = MockConfig()
        model = MockModel(config)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)
        assert intermediate_prompts.shape == (9, 2, 5, 16)
