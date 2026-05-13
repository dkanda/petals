import sys
import types
from unittest import mock

def test_ptune_shapes():
    import pytest
    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    mock_hivemind = types.ModuleType("hivemind")
    mock_hivemind.__path__ = []
    mock_hivemind.__spec__ = None
    mock_hivemind.get_logger = lambda name: None

    mock_petals_utils = types.ModuleType("petals.utils.misc")
    mock_petals_utils.__path__ = []
    mock_petals_utils.__spec__ = None
    mock_petals_utils.DUMMY = torch.empty(0)

    # We will mock the entire `petals` package and manually inject `petals.utils.misc`
    mock_petals = types.ModuleType("petals")
    mock_petals.__path__ = ["src/petals"]
    mock_petals.__spec__ = None

    mock_petals_u = types.ModuleType("petals.utils")
    mock_petals_u.__path__ = ["src/petals/utils"]
    mock_petals_u.__spec__ = None

    mock_petals_client = types.ModuleType("petals.client")
    mock_petals_client.__path__ = ["src/petals/client"]
    mock_petals_client.__spec__ = None

    mocks = {
        "hivemind": mock_hivemind,
        "petals.utils.misc": mock_petals_utils,
        "petals": mock_petals,
        "petals.utils": mock_petals_u,
        "petals.client": mock_petals_client,
    }

    with mock.patch.dict("sys.modules", mocks):
        from petals.client.ptune import PTuneMixin

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(100, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=4,
        )

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 16])
        assert intermediate_prompts.shape == torch.Size([4, 2, 5, 16])
