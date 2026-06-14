import sys
import os
sys.path.insert(0, os.path.abspath('src'))
from unittest import mock
import torch
from transformers import PretrainedConfig

mock_hivemind = mock.MagicMock()

with mock.patch.dict('sys.modules', {
    'hivemind': mock_hivemind,
    'hivemind.p2p': mock.MagicMock(),
    'hivemind.compression': mock.MagicMock(),
    'hivemind.moe': mock.MagicMock(),
    'hivemind.moe.client': mock.MagicMock(),
    'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
    'hivemind.utils': mock.MagicMock(),
    'hivemind.utils.logging': mock.MagicMock(),
    'hivemind.proto': mock.MagicMock(),
    'hivemind.proto.runtime_pb2': mock.MagicMock(),
    'tensor_parallel': mock.MagicMock(),
    'petals': mock.MagicMock(), # we just import the file directly below
}):
    # Import directly from the file to bypass __init__.py issues
    import importlib.util
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)

    # Needs to provide mock for petals.utils.misc
    misc_mock = mock.MagicMock()
    misc_mock.DUMMY = torch.empty(0)
    sys.modules['petals.utils'] = mock.MagicMock()
    sys.modules['petals.utils.misc'] = misc_mock

    spec.loader.exec_module(ptune)

class MockModel(ptune.PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = mock.MagicMock()
        self.word_embeddings.weight = mock.MagicMock()
        self.word_embeddings.weight.dtype = torch.float32
        self.word_embeddings.weight.device = torch.device('cpu')

        self.init_prompts(config)

def test_ptune_intermediate_prompts_shape():
    config = PretrainedConfig()
    config.tuning_mode = "deep_ptune"
    config.pre_seq_len = 5
    config.hidden_size = 64
    config.num_hidden_layers = 12

    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    # Assert Layer 0 prompts shape
    assert prompts.shape == torch.Size([2, 5, 64])

    # Assert intermediate prompts shape is for num_hidden_layers - 1
    assert intermediate_prompts.shape == torch.Size([11, 2, 5, 64])

def test_ptune_not_deep_ptune():
    config = PretrainedConfig()
    config.tuning_mode = "ptune"
    config.pre_seq_len = 5
    config.hidden_size = 64
    config.num_hidden_layers = 12

    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    # Assert Layer 0 prompts shape
    assert prompts.shape == torch.Size([2, 5, 64])

    # Assert intermediate prompts are DUMMY
    assert intermediate_prompts.shape == torch.Size([0])
