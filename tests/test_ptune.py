import sys
import os

sys.path.insert(0, os.path.abspath('src'))
from unittest import mock
import torch
import torch.nn as nn

def test_deep_ptune_intermediate_prompts_shape():
    petals_mock = mock.MagicMock()
    petals_mock.__path__ = []
    petals_mock.__spec__ = None

    hivemind_mock = mock.MagicMock()
    hivemind_mock.__path__ = []
    hivemind_mock.__spec__ = None

    transformers_mock = mock.MagicMock()
    transformers_mock.__path__ = []
    transformers_mock.__spec__ = None

    misc_mock = mock.MagicMock()
    misc_mock.DUMMY = torch.empty(0)

    mocks = {
        'hivemind': hivemind_mock,
        'transformers': transformers_mock,
        'petals': petals_mock,
        'petals.utils': mock.MagicMock(),
        'petals.utils.misc': misc_mock,
    }

    with mock.patch.dict('sys.modules', mocks):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["ptune"] = ptune
        spec.loader.exec_module(ptune)

        PTuneMixin = ptune.PTuneMixin

        class DummyConfig:
            def __init__(self):
                self.tuning_mode = "deep_ptune"
                self.pre_seq_len = 5
                self.hidden_size = 10
                self.num_hidden_layers = 4

        class DummyWordEmbeddings:
            weight = torch.empty(0, dtype=torch.float32)

        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = DummyWordEmbeddings()
                self.init_prompts(config)

        config = DummyConfig()
        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 5, 10)
        assert intermediate_prompts.shape == (3, 2, 5, 10)
