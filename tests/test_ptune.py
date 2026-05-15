import sys
import types
from unittest import mock
import torch
import torch.nn as nn
from dataclasses import dataclass

def test_ptune_intermediate_shape():
    @dataclass
    class PretrainedConfig:
        tuning_mode: str
        pre_seq_len: int
        hidden_size: int
        num_hidden_layers: int

    petals_mock = types.ModuleType('petals')
    petals_mock.__path__ = ["src/petals"]
    petals_mock.__spec__ = None

    petals_utils_mock = types.ModuleType('petals.utils')
    petals_utils_mock.__path__ = ["src/petals/utils"]
    petals_utils_mock.__spec__ = None
    petals_mock.utils = petals_utils_mock

    petals_utils_misc_mock = types.ModuleType('petals.utils.misc')
    petals_utils_misc_mock.DUMMY = torch.empty(0)
    petals_utils_mock.misc = petals_utils_misc_mock

    hivemind_mock = types.ModuleType('hivemind')
    def get_logger(name):
        import logging
        return logging.getLogger(name)
    hivemind_mock.get_logger = get_logger

    with mock.patch.dict('sys.modules', {
        'petals': petals_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
        'hivemind': hivemind_mock
    }):
        # Use importlib to bypass any parent package dependencies since we mocked sys.modules
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["ptune"] = ptune
        spec.loader.exec_module(ptune)

        class DummyModel(ptune.PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=16, num_hidden_layers=4)

        with mock.patch.object(ptune, '_original_register_parameter', nn.Module.register_parameter):
            model = DummyModel(config)

        prompts, intermediate = model.get_prompt(batch_size=2)
        assert prompts.shape == (2, 5, 16)
        assert intermediate.shape == (4, 2, 5, 16)