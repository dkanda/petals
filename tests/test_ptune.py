import torch
import sys
import importlib.util
from unittest import mock

def test_ptune_shapes():
    # Mock dependencies for isolated testing
    class MockPackage(mock.MagicMock):
        __path__ = []
        __spec__ = None

    mock_modules = {
        'hivemind': MockPackage(),
        'transformers': MockPackage(),
        'petals': MockPackage(),
        'petals.utils': MockPackage(),
        'petals.utils.misc': type(sys)('petals.utils.misc'),
    }
    mock_modules['petals.utils.misc'].DUMMY = torch.empty(0)

    # Load the module specifically from the file path
    filepath = "src/petals/client/ptune.py"
    spec = importlib.util.spec_from_file_location("ptune", filepath)
    ptune = importlib.util.module_from_spec(spec)

    with mock.patch.dict('sys.modules', mock_modules):
        sys.modules["ptune"] = ptune
        import torch.nn as nn
        _original_register_parameter = nn.Module.register_parameter
        spec.loader.exec_module(ptune)
        nn.Module.register_parameter = _original_register_parameter

    class MockConfig:
        def __init__(self):
            self.tuning_mode = "deep_ptune"
            self.pre_seq_len = 5
            self.hidden_size = 64
            self.num_hidden_layers = 10

    class MockWordEmbeddings:
        def __init__(self):
            self.weight = torch.zeros(1, dtype=torch.float32)

    class TestPTuneMixin(ptune.PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = MockWordEmbeddings()
            self.init_prompts(config)

    config = MockConfig()
    mixin = TestPTuneMixin(config)

    batch_size = 2
    prompts, intermediate_prompts = mixin.get_prompt(batch_size=batch_size)

    # Check that prompts output shape is correct
    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)

    # Check that intermediate_prompts shape is correct
    assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)

    # Ensure the first layer's intermediate prompt (index 0) is zero-padded
    assert torch.all(intermediate_prompts[0] == 0)

    # Also verify that tuning_mode="ptune" behaves correctly (returns dummy for intermediate prompts)
    config.tuning_mode = "ptune"
    mixin2 = TestPTuneMixin(config)
    prompts2, intermediate_prompts2 = mixin2.get_prompt(batch_size=batch_size)
    assert prompts2.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert (isinstance(intermediate_prompts2, torch.Tensor) and intermediate_prompts2.numel() == 0) or intermediate_prompts2 is sys.modules['petals.utils.misc'].DUMMY
