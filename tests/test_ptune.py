import sys
import importlib.util
from unittest import mock
import types
import torch
from transformers import PretrainedConfig

class MockPackage(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []

def test_ptune_intermediate_prompts_shape():
    mock_hivemind = MockPackage('hivemind')
    mock_hivemind.get_logger = mock.MagicMock()

    mock_petals = MockPackage('petals')
    mock_petals_utils = MockPackage('petals.utils')
    mock_petals_utils_misc = types.ModuleType('petals.utils.misc')
    mock_petals_utils_misc.DUMMY = torch.empty(0)

    with mock.patch.dict('sys.modules', {
        'hivemind': mock_hivemind,
        'petals': mock_petals,
        'petals.utils': mock_petals_utils,
        'petals.utils.misc': mock_petals_utils_misc
    }):
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune)

    class DummyModel(ptune.PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
            with mock.patch.object(ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
                self.init_prompts(config)

    config = PretrainedConfig(
        hidden_size=8,
        num_hidden_layers=4,
        tuning_mode="deep_ptune",
        pre_seq_len=5
    )
    model = DummyModel(config)
    batch_size = 2

    prompts, intermediate_prompts = model.get_prompt(batch_size)

    assert prompts.shape == (batch_size, 5, 8)
    assert intermediate_prompts.shape == (3, batch_size, 5, 8)
