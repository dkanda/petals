import sys
from unittest import mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import importlib.util

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def test_deep_ptune():
    sys_modules_mock = {
        'petals': MockPackage(),
        'hivemind': mock.MagicMock(),
        'hivemind.get_logger': mock.MagicMock()
    }

    # We also need to provide DUMMY in petals.utils.misc
    misc_module = type(sys)('petals.utils.misc')
    misc_module.DUMMY = torch.zeros(1)
    sys_modules_mock['petals.utils.misc'] = misc_module

    with mock.patch.dict('sys.modules', sys_modules_mock):
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune)

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                # Mock word_embeddings to provide weight and device
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight = torch.zeros(1, dtype=torch.float32)
                self.init_prompts(config)

        config = PretrainedConfig()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 5
        config.hidden_size = 16
        config.num_hidden_layers = 4

        model = DummyModel(config)

        # Test get_prompt
        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        assert prompts.shape == (batch_size, 5, 16)
        assert intermediate_prompts.shape == (4, batch_size, 5, 16)

        # Check first layer is zeros
        assert torch.all(intermediate_prompts[0] == 0)

        # Check remaining layers are not just zeros (from normal initialization)
        assert not torch.all(intermediate_prompts[1] == 0)

if __name__ == "__main__":
    test_deep_ptune()
    print("Test passed!")
