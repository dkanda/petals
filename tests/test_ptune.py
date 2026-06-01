import sys
import os
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from unittest import mock

sys.path.insert(0, os.path.abspath("src"))

def test_ptune_shape():
    # Mock hivemind to avoid ModuleNotFoundError
    hivemind_mock = mock.MagicMock()
    hivemind_mock.get_logger = mock.MagicMock(return_value=mock.MagicMock())

    # Mock petals to avoid import issues
    petals_mock = mock.MagicMock()

    # Need to mock petals.utils.misc.DUMMY inside ptune
    petals_utils_misc_mock = mock.MagicMock()
    petals_utils_misc_mock.DUMMY = torch.empty(0)

    with mock.patch.dict('sys.modules', {
        'hivemind': hivemind_mock,
        'petals': petals_mock,
        'petals.utils.misc': petals_utils_misc_mock,
    }):
        import importlib.util
        spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.init_prompts(config)
                self.word_embeddings = type('obj', (object,), {})()
                self.word_embeddings.weight = torch.empty(0, dtype=torch.float32)

        config = PretrainedConfig(hidden_size=8, num_hidden_layers=5)
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 4

        model = DummyModel(config)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert prompts.shape == torch.Size([2, 4, 8])
        assert intermediate_prompts is not petals_utils_misc_mock.DUMMY
        assert intermediate_prompts.shape == torch.Size([4, 2, 4, 8])
