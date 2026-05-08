import sys
import unittest.mock as mock
import importlib.util

def test_ptune_deep_ptune():
    class MockPackage:
        pass

    import torch
    import torch.nn as nn

    with mock.patch.dict('sys.modules', {
        'petals': MockPackage(),
        'petals.utils': MockPackage(),
        'petals.utils.misc': MockPackage(),
        'hivemind': MockPackage(),
        'transformers': MockPackage()
    }):

        sys.modules['petals.utils.misc'].DUMMY = torch.empty(0)
        sys.modules['hivemind'].get_logger = lambda name: None
        sys.modules['transformers'].PretrainedConfig = object

        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune)

        class DummyConfig:
            def __init__(self):
                self.pre_seq_len = 10
                self.tuning_mode = "deep_ptune"
                self.hidden_size = 32
                self.num_hidden_layers = 4

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.init_prompts(config)
                self.word_embeddings = type('DummyEmbed', (), {'weight': torch.randn(10, config.hidden_size)})()

        sys.modules['petals.client'] = MockPackage()
        sys.modules['petals.client.ptune'] = ptune
        with mock.patch('petals.client.ptune._original_register_parameter', torch.nn.Module.register_parameter):
            config = DummyConfig()
            model = DummyModel(config)

            p, ip = model.get_prompt(2)
            assert p.shape == (2, 10, 32)
            assert ip.shape == (4, 2, 10, 32)
            assert torch.allclose(ip[0], p)
