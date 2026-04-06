import sys
import importlib.util
from unittest import mock
import torch

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def setup_mocks():
    mock_dict = {
        'hivemind': MockPackage(),
        'hivemind.moe': MockPackage(),
        'hivemind.moe.client': MockPackage(),
        'hivemind.moe.client.remote_expert_worker': MockPackage(),
        'transformers': MockPackage(),
        'petals.utils': MockPackage(),
        'petals.utils.misc': MockPackage(),
    }
    mock_dict['petals'] = MockPackage()
    return mock.patch.dict(sys.modules, mock_dict)

def load_ptune_module():
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)

    # Mock DUMMY tensor
    DUMMY = torch.zeros(1)

    # Inject mocked dependencies into module namespace
    ptune.DUMMY = DUMMY
    ptune.nn = torch.nn
    ptune.torch = torch

    class MockConfig:
        def __init__(self, tuning_mode, pre_seq_len, hidden_size, num_hidden_layers=None):
            self.tuning_mode = tuning_mode
            self.pre_seq_len = pre_seq_len
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers

        def __bool__(self):
            return True

        # Add magic methods for mock compatibility
        def __getattr__(self, name):
            if name in ['tuning_mode', 'pre_seq_len', 'hidden_size', 'num_hidden_layers']:
                return self.__dict__[name]
            raise AttributeError(name)

    ptune.PretrainedConfig = MockConfig

    # Pre-inject to sys.modules
    with setup_mocks():
        sys.modules['petals.utils.misc'].DUMMY = DUMMY
        spec.loader.exec_module(ptune)

    # Actually monkeypatch the imported PretrainedConfig inside the loaded module
    # because it might have been overwritten during exec_module
    ptune.PretrainedConfig = MockConfig

    return ptune, DUMMY

def test_ptune():
    ptune, DUMMY = load_ptune_module()

    mixin = ptune.PTuneMixin()
    mixin.config = ptune.PretrainedConfig(tuning_mode="ptune", pre_seq_len=2, hidden_size=8)
    # The config argument passed to init_prompts must have tuning_mode etc.
    # In MockPackage environment, type checking might fail, so we just pass our MockConfig
    mixin.init_prompts(mixin.config)

    # Mock word_embeddings to provide weight.device and weight.dtype
    mixin.word_embeddings = mock.MagicMock()
    mixin.word_embeddings.weight.device = torch.device('cpu')
    mixin.word_embeddings.weight.dtype = torch.float32

    prompts, intermediate_prompts = mixin.get_prompt(batch_size=3)

    # In ptune mode, intermediate_prompts should be DUMMY
    assert "DUMMY.to()" in repr(intermediate_prompts) or intermediate_prompts is DUMMY or torch.equal(intermediate_prompts, DUMMY.to(torch.float32))

def test_deep_ptune():
    ptune, DUMMY = load_ptune_module()

    mixin = ptune.PTuneMixin()
    mixin.config = ptune.PretrainedConfig(tuning_mode="deep_ptune", pre_seq_len=2, hidden_size=8, num_hidden_layers=4)
    mixin.init_prompts(mixin.config)

    # Mock word_embeddings to provide weight.device and weight.dtype
    mixin.word_embeddings = mock.MagicMock()
    mixin.word_embeddings.weight.device = torch.device('cpu')
    mixin.word_embeddings.weight.dtype = torch.float32

    prompts, intermediate_prompts = mixin.get_prompt(batch_size=3)

    # In deep_ptune mode, shape should be (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (4, 3, 2, 8)

    # Verify zero padding on the first layer
    assert torch.all(intermediate_prompts[0] == 0)
