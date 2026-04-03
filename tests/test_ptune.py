import sys
import importlib.util
from unittest import mock
import torch
import pytest

# Mock package class to fake nested modules
class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

_original_modules = {}

def setup_module():
    global _original_modules
    _original_modules['hivemind'] = sys.modules.get('hivemind')
    _original_modules['petals'] = sys.modules.get('petals')
    _original_modules['petals.client.ptune'] = sys.modules.get('petals.client.ptune')
    _original_modules['petals.utils.misc'] = sys.modules.get('petals.utils.misc')

    sys.modules['hivemind'] = MockPackage()
    sys.modules['petals'] = MockPackage()

    # Load petals.client.ptune module directly to avoid petals/__init__.py imports
    spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
    ptune_module = importlib.util.module_from_spec(spec)
    sys.modules["petals.client.ptune"] = ptune_module

    # Mock petals.utils.misc.DUMMY
    sys.modules["petals.utils.misc"] = mock.MagicMock()
    sys.modules["petals.utils.misc"].DUMMY = "DUMMY_TENSOR"
    spec.loader.exec_module(ptune_module)

    global PTuneMixin
    PTuneMixin = ptune_module.PTuneMixin

def teardown_module():
    global _original_modules
    for module_name, original_module in _original_modules.items():
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module

def test_ptune_intermediate_shape():
    from transformers import PretrainedConfig

    batch_size = 2
    pre_seq_len = 5
    hidden_size = 10
    num_hidden_layers = 3

    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = pre_seq_len
    config.hidden_size = hidden_size
    config.num_hidden_layers = num_hidden_layers

    class MockModel(PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = mock.MagicMock()
            self.word_embeddings.weight.device = torch.device("cpu")
            self.word_embeddings.weight.dtype = torch.float32
            self.init_prompts(config)

    model = MockModel(config)

    # Verify intermediate embedding parameters shape
    expected_dim_size = (num_hidden_layers - 1) * hidden_size
    assert model.intermediate_prompt_embeddings.weight.shape == (pre_seq_len, expected_dim_size), \
        f"Expected intermediate embedding dim to be {expected_dim_size}, got {model.intermediate_prompt_embeddings.weight.shape[1]}"

    # Get prompts and verify intermediate shapes
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Prompts shape should be [batch_size, pre_seq_len, hidden_size]
    assert prompts.shape == (batch_size, pre_seq_len, hidden_size)

    # intermediate_prompts shape should be [num_hidden_layers, batch_size, pre_seq_len, hidden_size]
    assert intermediate_prompts.shape == (num_hidden_layers, batch_size, pre_seq_len, hidden_size), \
        f"Expected intermediate_prompts shape to be {(num_hidden_layers, batch_size, pre_seq_len, hidden_size)}, got {intermediate_prompts.shape}"

    # Check that the first layer's tensor contains all zeros
    first_layer_prompts = intermediate_prompts[0]
    assert torch.all(first_layer_prompts == 0), "The first layer's intermediate prompt padding should be all zeros"

    # Check that the rest of the layers are not all zeros (assuming random init)
    rest_layer_prompts = intermediate_prompts[1:]
    assert not torch.all(rest_layer_prompts == 0), "The subsequent layers should contain embeddings, not just zeros"
