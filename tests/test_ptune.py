import sys
import os
from unittest import mock
import torch
import pytest
from transformers import PretrainedConfig

sys.path.insert(0, os.path.abspath('src'))

# Prepare mocks
hivemind_mock = mock.MagicMock()
hivemind_mock.utils = mock.MagicMock()
hivemind_mock.p2p = mock.MagicMock()
hivemind_mock.moe = mock.MagicMock()
hivemind_mock.moe.client = mock.MagicMock()
hivemind_mock.dht = mock.MagicMock()

mocks = {
    'hivemind': hivemind_mock,
    'hivemind.utils': hivemind_mock.utils,
    'hivemind.utils.logging': mock.MagicMock(),
    'hivemind.utils.logging.get_logger': mock.MagicMock(),
    'hivemind.p2p': hivemind_mock.p2p,
    'hivemind.moe': hivemind_mock.moe,
    'hivemind.moe.client': hivemind_mock.moe.client,
    'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
    'hivemind.moe.expert_uid': mock.MagicMock(),
    'hivemind.dht': hivemind_mock.dht,
    'hivemind.compression': mock.MagicMock(),
    'hivemind.compression.base': mock.MagicMock(),
    'tensor_parallel': mock.MagicMock(),
    'petals.client': mock.MagicMock(__path__=["src/petals/client"], __spec__=None),
    'petals.client.inference_session': mock.MagicMock(),
    'petals.client.remote_sequential': mock.MagicMock(),
    'petals.client.routing': mock.MagicMock(),
    'petals.models': mock.MagicMock(__path__=["src/petals/models"], __spec__=None),
}

def test_ptune_intermediate_prompts_shape():
    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin
        from petals.utils.misc import DUMMY

        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32

                # Mock parameter registration behavior in init_empty_weights context
                with mock.patch("torch.nn.Module.register_parameter", new=mock.MagicMock()):
                    self.init_prompts(config)
                    # For tests, the nn.Embedding creates meta tensors if not careful, but the scratchpad showed we need to actually
                    # avoid errors with Embedding init. But since we actually didn't need to patch in the final fix script
                    # we can just call it normally because PTuneMixin does not call register_parameter itself, only within context managers
                    pass

        # Since force_non_empty_weights context manager patches register_parameter,
        # let's just make sure it doesn't crash
        config = PretrainedConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=64, num_hidden_layers=10)

        # Instantiate DummyModel directly
        class DummyModelSimple(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32
                self.init_prompts(config)

        model = DummyModelSimple(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert intermediate_prompts.shape == (9, 2, 5, 64)
