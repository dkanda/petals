import sys
import os
import importlib
sys.path.insert(0, os.path.abspath('src'))
import pytest
import torch
import torch.nn as nn
from unittest import mock

def test_ptune_intermediate_prompts_shape():
    from transformers import PretrainedConfig

    class MockModule:
        __path__ = []
        __spec__ = None

    with mock.patch.dict('sys.modules', {
        'hivemind': mock.MagicMock()
    }):
        spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune

        class MockMisc:
            DUMMY = torch.empty(0)
            __path__ = []
        petals_utils_misc = MockMisc()
        sys.modules["petals.utils"] = MockMisc()
        sys.modules["petals.utils.misc"] = petals_utils_misc

        spec.loader.exec_module(ptune)

        PTuneMixin = ptune.PTuneMixin

        class DummyModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=4
        )

        with mock.patch.object(ptune, '_original_register_parameter', nn.Module.register_parameter):
            model = DummyModel(config)

            assert model.prompt_embeddings.weight.shape == (5, 16)
            assert model.intermediate_prompt_embeddings.weight.shape == (5, (4 - 1) * 16)

            prompts, intermediate_prompts = model.get_prompt(batch_size=2)
            assert prompts.shape == (2, 5, 16)
            assert intermediate_prompts.shape == (3, 2, 5, 16)

if __name__ == "__main__":
    test_ptune_intermediate_prompts_shape()
