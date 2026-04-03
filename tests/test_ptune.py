import sys
from unittest import mock
import importlib.util

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

with mock.patch.dict(
    "sys.modules",
    {
        "petals": MockPackage(), # Mock petals to avoid __init__.py loading hivemind
        "petals.utils": MockPackage(),
        "petals.utils.misc": MockPackage(),
        "hivemind": MockPackage(),
        "hivemind.get_logger": mock.MagicMock(),
    },
):
    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    # Needs a real DUMMY for testing because it sets .to(dtype)
    sys.modules["petals.utils.misc"].DUMMY = torch.zeros(1)

    # Load module directly
    spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
    ptune_module = importlib.util.module_from_spec(spec)
    sys.modules["petals.client.ptune"] = ptune_module
    spec.loader.exec_module(ptune_module)

    PTuneMixin = ptune_module.PTuneMixin

class MockModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4
    )

    model = MockModel(config)

    # Verify embeddings initialized correctly
    assert hasattr(model, 'prompt_embeddings')
    assert hasattr(model, 'intermediate_prompt_embeddings')

    # Prompt embeddings
    assert model.prompt_embeddings.weight.shape == (5, 16)

    # Intermediate prompt embeddings
    assert model.intermediate_prompt_embeddings.weight.shape == (5, (4 - 1) * 16)

    # Verify get_prompt
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Final shapes
    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (4, 2, 5, 16)

    # Ensure first layer is zero-padded
    first_layer_prompts = intermediate_prompts[0]
    assert torch.all(first_layer_prompts == 0)
