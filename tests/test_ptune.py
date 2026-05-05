import sys
import unittest.mock
import torch
from transformers import PretrainedConfig

class MockMisc:
    DUMMY = torch.tensor([0.0])

mock_misc = MockMisc()

import contextlib
import importlib.util

@contextlib.contextmanager
def setup_mock_env():
    # To avoid the massive import tree required for a simple unit test,
    # we isolate the module exactly how `sys.modules` patching intended
    # but do it fully safely using importlib rather than direct `import` which triggers __init__.py

    import types
    mock_hivemind = unittest.mock.MagicMock()
    mock_petals_misc = mock_misc

    with unittest.mock.patch.dict(sys.modules, {
        'hivemind': mock_hivemind,
        'petals.utils.misc': mock_petals_misc
    }):
        spec = importlib.util.spec_from_file_location("petals_client_ptune", "src/petals/client/ptune.py")
        ptune_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune_module)
        yield ptune_module

def test_ptune_dimensions():
    with setup_mock_env() as ptune:
        PTuneMixin = ptune.PTuneMixin

        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        with unittest.mock.patch.object(torch.nn.Module, 'register_parameter', ptune._original_register_parameter):
            config = PretrainedConfig(tuning_mode="deep_ptune", pre_seq_len=5, num_hidden_layers=10, hidden_size=16)
            model = DummyModel(config)
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)
            assert prompts.shape == torch.Size([2, 5, 16])
            assert intermediate_prompts.shape == torch.Size([9, 2, 5, 16])
