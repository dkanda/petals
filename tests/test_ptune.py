import sys
import importlib.util
from unittest import mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def setup_module():
    sys.modules['hivemind'] = MockPackage()

def teardown_module():
    if 'hivemind' in sys.modules:
        del sys.modules['hivemind']

def test_deep_ptune():
    with mock.patch.dict(sys.modules, {'hivemind': MockPackage()}):
        spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune

        spec2 = importlib.util.spec_from_file_location("petals.utils.misc", "src/petals/utils/misc.py")
        misc = importlib.util.module_from_spec(spec2)
        sys.modules["petals.utils.misc"] = misc
        misc.DUMMY = torch.empty(0)
        spec2.loader.exec_module(misc)

        spec.loader.exec_module(ptune)

        PTuneMixin = ptune.PTuneMixin

        class MockModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=8,
            num_hidden_layers=4,
        )
        model = MockModel(config)
        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

        assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
        assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

        # Verify first layer is zero padding
        assert torch.all(intermediate_prompts[0] == 0)
