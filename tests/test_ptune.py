import torch
import pytest
from transformers import PretrainedConfig
import sys
from unittest import mock
import importlib.util

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def test_deep_ptune():
    with mock.patch.dict(
        sys.modules,
        {
            "petals": MockPackage(),
            "hivemind": MockPackage(),
            "petals.utils": MockPackage(),
            "petals.utils.misc": mock.MagicMock(),
        },
    ):
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)

        import petals.utils.misc
        ptune.DUMMY = mock.MagicMock()
        sys.modules['ptune'] = ptune
        spec.loader.exec_module(ptune)

        PTuneMixin = ptune.PTuneMixin

        class MockWordEmbeddings:
            def __init__(self):
                self.weight = torch.zeros(1, dtype=torch.float32, device='cpu')

        class MockModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = MockWordEmbeddings()
                self.init_prompts(config)

        config = PretrainedConfig()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 5
        config.hidden_size = 16
        config.num_hidden_layers = 4

        model = MockModel(config)

        # Check weight shape
        assert model.prompt_embeddings.weight.shape == (5, 16)
        assert model.intermediate_prompt_embeddings.weight.shape == (5, 3 * 16)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 16])
        assert intermediate_prompts.shape == torch.Size([4, 2, 5, 16])

        # Verify the prepended padding
        assert torch.all(intermediate_prompts[0] == 0)
