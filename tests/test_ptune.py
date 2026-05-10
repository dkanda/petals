import torch
import torch.nn as nn
from unittest import mock
import sys
import importlib.util

class MockPackage:
    __path__ = []
    __spec__ = None

class MockConfig:
    def __init__(self, tuning_mode, pre_seq_len, hidden_size, num_hidden_layers):
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

def test_ptune_mixin_deep_ptune():
    with mock.patch.dict('sys.modules', {
        'hivemind': mock.MagicMock(),
        'transformers': mock.MagicMock(),
        'petals': MockPackage(),
        'petals.utils': MockPackage(),
        'petals.utils.misc': mock.MagicMock(),
    }):
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

        class MockModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = MockConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=4,
            hidden_size=16,
            num_hidden_layers=3
        )

        with mock.patch.object(ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
            model = MockModel(config)

            assert model.prompt_embeddings.weight.shape == (4, 16)
            assert model.intermediate_prompt_embeddings.weight.shape == (4, 2 * 16)

            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            assert prompts.shape == (batch_size, 4, 16)
            assert intermediate_prompts.shape == (3, batch_size, 4, 16)
            assert torch.allclose(intermediate_prompts[0], prompts)
