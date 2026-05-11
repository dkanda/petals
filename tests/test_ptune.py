import torch
import torch.nn as nn
from unittest import mock
import sys
import types

# Test specifically for PTuneMixin shapes (num_hidden_layers - 1 fix).
# To avoid missing dependencies breaking the test suite, we mock necessary components.

def setup_mocks():
    hivemind_mock = types.ModuleType("hivemind")
    hivemind_mock.__path__ = []
    hivemind_mock.__spec__ = None
    hivemind_mock.get_logger = mock.MagicMock()

    transformers_mock = types.ModuleType("transformers")
    transformers_mock.__path__ = []
    transformers_mock.__spec__ = None
    transformers_mock.PretrainedConfig = mock.MagicMock

    petals_mock = types.ModuleType("petals")
    petals_mock.__path__ = []
    petals_mock.__spec__ = None

    petals_utils_mock = types.ModuleType("petals.utils")
    petals_utils_mock.__path__ = []
    petals_utils_mock.__spec__ = None

    petals_utils_misc_mock = types.ModuleType("petals.utils.misc")
    petals_utils_misc_mock.DUMMY = "DUMMY"

    sys_modules_mocks = {
        "hivemind": hivemind_mock,
        "transformers": transformers_mock,
        "petals": petals_mock,
        "petals.utils": petals_utils_mock,
        "petals.utils.misc": petals_utils_misc_mock,
    }
    return sys_modules_mocks

def test_ptune_deep_ptune_prompts():
    mocks = setup_mocks()
    with mock.patch.dict("sys.modules", mocks):
        # We need importlib to avoid triggering the whole petals module structure loading
        # which imports too many missing dependencies
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune)

        with mock.patch.object(ptune, "_original_register_parameter", nn.Module.register_parameter):
            class DummyConfig:
                tuning_mode = "deep_ptune"
                pre_seq_len = 5
                hidden_size = 16
                num_hidden_layers = 4

            class DummyWordEmbeddings:
                weight = nn.Parameter(torch.empty(1, 1, dtype=torch.float32))

            class DummyModel(ptune.PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = DummyWordEmbeddings()
                    self.init_prompts(config)

            config = DummyConfig()
            model = DummyModel(config)

            assert model.prompt_embeddings.weight.shape == (5, 16)
            assert model.intermediate_prompt_embeddings.weight.shape == (5, (4 - 1) * 16)

            prompts, intermediate_prompts = model.get_prompt(batch_size=2)
            assert prompts.shape == (2, 5, 16)
            assert intermediate_prompts.shape == (3, 2, 5, 16)
