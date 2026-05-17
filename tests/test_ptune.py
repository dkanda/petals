import sys
import types
from unittest import mock
import torch

def test_ptune_mixin_deep_ptune():
    sys_modules_patch = {
        'hivemind': mock.MagicMock(),
        'transformers': mock.MagicMock(),
        'petals': mock.MagicMock(__path__=[], __spec__=None),
        'petals.utils': mock.MagicMock(__path__=[], __spec__=None),
        'petals.utils.misc': mock.MagicMock(__path__=[], __spec__=None),
    }
    sys_modules_patch['petals.utils.misc'].DUMMY = torch.empty(0)

    with mock.patch.dict('sys.modules', sys_modules_patch):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules['petals.client.ptune'] = ptune
        spec.loader.exec_module(ptune)

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = type('WE', (), {'weight': type('W', (), {'device': torch.device('cpu'), 'dtype': torch.float32})()})()

        config = type('Config', (), {})
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 4
        config.hidden_size = 8
        config.num_hidden_layers = 3

        model = DummyModel(config)
        model.init_prompts(config)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)
        assert intermediate_prompts.shape == (2, 2, 4, 8)
