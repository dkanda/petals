import sys
import types
from unittest import mock

import torch
import torch.nn as nn

mocks = {
    'hivemind': types.ModuleType('hivemind'),
    'transformers': types.ModuleType('transformers'),
    'petals': types.ModuleType('petals'),
    'petals.utils': types.ModuleType('petals.utils'),
    'petals.utils.misc': types.ModuleType('petals.utils.misc'),
}
mocks['hivemind'].__path__ = []
mocks['hivemind'].__spec__ = None
mocks['hivemind'].get_logger = lambda name: mock.MagicMock()

mocks['transformers'].__path__ = []
mocks['transformers'].__spec__ = None
mocks['transformers'].PretrainedConfig = type('PretrainedConfig', (), {})

mocks['petals'].__path__ = []
mocks['petals'].__spec__ = None

mocks['petals.utils'].__path__ = []
mocks['petals.utils'].__spec__ = None

mocks['petals.utils.misc'].__path__ = []
mocks['petals.utils.misc'].__spec__ = None
mocks['petals.utils.misc'].DUMMY = mock.MagicMock()

def test_ptune_intermediate_prompts_shape():
    with mock.patch.dict('sys.modules', mocks):
        with open("src/petals/client/ptune.py", "r") as f:
            ptune_src = f.read()

        ptune_module = types.ModuleType("ptune_module")

        ptune_module.__dict__["torch"] = torch
        ptune_module.__dict__["nn"] = nn
        ptune_module.__dict__["dataclasses"] = __import__("dataclasses")
        ptune_module.__dict__["contextmanager"] = __import__("contextlib").contextmanager
        ptune_module.__dict__["Optional"] = __import__("typing").Optional

        exec(ptune_src, ptune_module.__dict__)

        PTuneMixin = ptune_module.PTuneMixin

        class DummyConfig:
            def __init__(self, pre_seq_len=5, tuning_mode="deep_ptune", hidden_size=8, num_hidden_layers=3):
                self.pre_seq_len = pre_seq_len
                self.tuning_mode = tuning_mode
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers

        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)

        model = DummyModel(DummyConfig())
        model.init_prompts(model.config)

        assert model.intermediate_prompt_embeddings.weight.shape == torch.Size([5, 16])

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)
        assert prompts.shape == torch.Size([2, 5, 8])
        assert intermediate_prompts.shape == torch.Size([2, 2, 5, 8])

if __name__ == "__main__":
    test_ptune_intermediate_prompts_shape()
