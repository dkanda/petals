import pytest
import torch
import sys
from transformers import PretrainedConfig
from unittest import mock
import types

def test_ptune_intermediate_prompts_shape():
    mocks = {
        'petals.utils.misc': mock.MagicMock(),
        'hivemind': mock.MagicMock(),
    }
    mocks['petals.utils.misc'].DUMMY = torch.empty(0)
    mocks['hivemind'].get_logger = mock.MagicMock()

    with mock.patch.dict('sys.modules', mocks):
        with open('src/petals/client/ptune.py', 'r') as f:
            src = f.read()

        module = types.ModuleType('ptune')
        module.__dict__['torch'] = torch
        module.__dict__['nn'] = torch.nn
        exec(src, module.__dict__)

        PTuneMixin = module.PTuneMixin

        class MockModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = torch.nn.Embedding(1, 1)
                self.word_embeddings.weight = torch.nn.Parameter(torch.empty(1, dtype=torch.float32))

        config = PretrainedConfig()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 16
        config.hidden_size = 32
        config.num_hidden_layers = 10

        model = MockModel(config)
        model.init_prompts(config)

        assert model.intermediate_prompt_embeddings.weight.shape == (16, (10 - 1) * 32)

        prompts, intermediate_prompts = model.get_prompt(batch_size=4)
        assert intermediate_prompts.shape == (10, 4, 16, 32)

if __name__ == '__main__':
    pytest.main(['-v', 'tests/test_ptune.py'])
