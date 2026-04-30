import sys
from unittest import mock
import torch
import torch.nn as nn
import importlib.util

class Config:
    def __init__(self, tuning_mode):
        self.num_hidden_layers = 10
        self.hidden_size = 64
        self.tuning_mode = tuning_mode
        self.pre_seq_len = 5

def _load_mocked_ptune():
    mock_modules = {
        'hivemind': mock.MagicMock(),
        'transformers': mock.MagicMock(),
        'petals': mock.MagicMock(),
        'petals.utils': mock.MagicMock(),
        'petals.utils.misc': mock.MagicMock(),
    }
    with mock.patch.dict('sys.modules', mock_modules):
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["src.petals.client.ptune"] = ptune
        ptune._original_register_parameter = nn.Module.register_parameter
        spec.loader.exec_module(ptune)
    return ptune

def test_ptune_shapes_deep_ptune():
    ptune = _load_mocked_ptune()

    class MockModel(ptune.PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = mock.MagicMock()
            self.word_embeddings.weight.device = "cpu"
            self.word_embeddings.weight.dtype = torch.float32

    with mock.patch.object(ptune, 'DUMMY', torch.empty(0)):
        config = Config("deep_ptune")
        model = MockModel(config)
        model.init_prompts(config)

        assert model.intermediate_prompt_embeddings.weight.shape == (5, 9 * 64)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        assert prompts.shape == (2, 5, 64)
        assert intermediate_prompts.shape == (10, 2, 5, 64)

def test_ptune_shapes_ptune():
    ptune = _load_mocked_ptune()

    class MockModel(ptune.PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = mock.MagicMock()
            self.word_embeddings.weight.device = "cpu"
            self.word_embeddings.weight.dtype = torch.float32

    with mock.patch.object(ptune, 'DUMMY', torch.empty(0)):
        config = Config("ptune")
        model = MockModel(config)
        model.init_prompts(config)

        assert not hasattr(model, 'intermediate_prompt_embeddings')

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        assert prompts.shape == (2, 5, 64)
        assert isinstance(intermediate_prompts, torch.Tensor)
        assert intermediate_prompts.numel() == 0
