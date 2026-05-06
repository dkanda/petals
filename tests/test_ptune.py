import sys
import types
from unittest import mock
import torch
import torch.nn as nn

def test_ptune_shapes():
    mock_petals = types.ModuleType("petals")
    mock_petals.utils = types.ModuleType("petals.utils")
    mock_petals.utils.misc = types.ModuleType("petals.utils.misc")
    mock_petals.utils.misc.DUMMY = torch.empty(0)

    with mock.patch.dict("sys.modules", {
        "petals": mock_petals,
        "petals.utils": mock_petals.utils,
        "petals.utils.misc": mock_petals.utils.misc,
        # mock hivemind if needed
        "hivemind": types.ModuleType("hivemind"),
    }):
        import hivemind
        hivemind.get_logger = lambda name: None

        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

        with mock.patch.object(ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
            class DummyConfig:
                tuning_mode = "deep_ptune"
                pre_seq_len = 5
                num_hidden_layers = 4
                hidden_size = 16

            class DummyModel(nn.Module, ptune.PTuneMixin):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, 16)
                    self.init_prompts(config)

            config = DummyConfig()
            model = DummyModel(config)

            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert prompts.shape == torch.Size([2, 5, 16])
            assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16])
